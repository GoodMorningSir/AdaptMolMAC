"""Dynamic multi-signal decoding for overlapping molecular channels.

This module maintains channel-wise prediction states, estimates bits under
interference, and detects newly emerging signals when residual errors become
abnormal.
"""

import numpy as np
from sklearn.metrics import mean_squared_error
from itertools import product
import uuid
import copy
from collections import deque
import warnings
import multiprocessing as mp
import sys
import shutil

from ..models import yRxData, ChanneModel_Rx
from ..mcutils import TProbAnomalyDetector, Plotter, logger, BinaryUtils
from ..viterbi import hypothesis_extended_viterbi_decode
from ..config import Settings
from .signal_processor import SignalProcessor
from .stationary_detector import StationaryProcessor



class ChannelPredictionItem:
    """Track one channel's decoded bits and running waveform prediction.

    Attributes:
        id (int): Instance identifier used in logs.
        prev_state_id (str | None): Saved decoder state used for rollback.
    """

    next_id = 0

    def __init__(self, chnl_info: ChanneModel_Rx, start_pos, yRx_len):
        """Create a prediction state for one estimated receive channel.

        Args:
            chnl_info (ChanneModel_Rx): Fitted channel template.
            start_pos (int): Starting sample index for waveform synthesis.
            yRx_len (int): Length of the full received signal buffer.
        """
        
        self.id = ChannelPredictionItem.next_id
        ChannelPredictionItem.next_id += 1

        self.__chnl_info = chnl_info
        self.__interval = chnl_info.interval
        self.__start_pos = start_pos
        self.__bits = ''
        self.__abnormal_detector = TProbAnomalyDetector()
        logger.log(f"[ChannelPredictionItem#{self.id}] Initialized with channel_model={chnl_info.id}")
        
        self.__pred_end = start_pos
        self.__pred = np.zeros(yRx_len, dtype=np.float64)
        
        self.prev_state_id = None

    def reset_abnormal_detector(self):
        """Clear the channel-specific anomaly history."""
        self.__abnormal_detector.reset()
        logger.log(f"[ChannelPredictionItem#{self.id}] Abnormal detector reset")
        
    def generate_bit_waveform(self, bit):
        """Delegate waveform generation to the underlying channel model."""
        return self.__chnl_info.generate_bit_waveform(bit)
    
    @property
    def n_preamble(self) -> int:
        return self.__chnl_info.n_preamble
    
    @property
    def preamble_bits(self) -> str:
        return self.__chnl_info.preamble_bits
    
    @property
    def pred(self) -> np.ndarray:
        return self.__pred
    
    @property
    def bits(self) -> str:
        return self.__bits
    
    @property
    def interval(self) -> int:
        return self.__interval
    
    @property
    def pred_end(self) -> int:
        return self.__pred_end
    
    @pred_end.setter
    def pred_end(self, value: int):
        self.__pred_end = value
        
    def add_err(self, error: float):
        """Append one reconstruction error sample to the anomaly detector."""
        self.__abnormal_detector.add(error)
        
    def detect_err(self, error):
        """Score an error sample and report whether it looks abnormal."""
        is_abnormal, kl_value = self.__abnormal_detector.detect(error)
        
        # if not is_abnormal:
        #     self.__abnormal_detector.add(error)
            
        return is_abnormal, kl_value

    def update_bit(self, bit):
        """Append one decoded bit and extend the predicted waveform.

        Args:
            bit (int): Newly decoded bit value.

        Returns:
            bool: Always returns True after the prediction state is updated.

        Raises:
            AssertionError: If `bit` is not 0 or 1.
        """
        
        assert (bit in [0, 1]), "bit must be 0 or 1"
        
        self.__bits += str(bit)
        bit_wave = self.__chnl_info.generate_bit_waveform(bit)
        self.__pred[self.__pred_end:self.__pred_end + len(bit_wave)] += bit_wave
        self.__pred_end += self.__interval

        return True
    
    def generate_waveform_from_bits(self, bit_sequence: str) -> np.ndarray:
        """Generate a waveform prediction for an arbitrary bit sequence.

        Args:
            bit_sequence (str): Bit sequence to synthesize from the channel
                template.

        Returns:
            numpy.ndarray: Predicted waveform aligned to the full receive
            buffer.
        """
        new_wave = np.zeros_like(self.__pred)
        current_pos = self.__start_pos
        for bit_char in bit_sequence:
            bit = int(bit_char)
            assert bit in ('0', '1', 0 ,1), f"Invalid bit value: {bit}"
            bit_wave = self.__chnl_info.generate_bit_waveform(bit)
            wave_length = len(bit_wave)
            end_pos = current_pos + wave_length
            if current_pos >= len(new_wave):
                break
            if end_pos > len(new_wave):
                bit_wave = bit_wave[:len(new_wave) - current_pos]
                end_pos = len(new_wave)
            new_wave[current_pos:end_pos] += bit_wave
            current_pos += self.__interval
            if current_pos >= len(new_wave):
                break
            
        return new_wave


class DynamicDecoder:
    """Decode overlapping channels while detecting newly appearing signals.

    The decoder iteratively estimates the next bit for each active channel,
    monitors residual errors for anomalies, and fits new channels when the
    current explanation of the received signal is insufficient.

    Attributes:
        id (int): Instance identifier used in logs.
        chnl_pred_list (list[ChannelPredictionItem]): Active channel states.
        signal_processor (SignalProcessor): Residual preprocessing helper.
        stationary_processor (StationaryProcessor): New-signal estimator.
    """

    next_id = 0
    MAX_TEMP_STATES = 50

    def __init__(self, yRx, chnl_info_list: list[ChanneModel_Rx], generator=None):
        """Initialize the decoder from a received signal and fitted channels.

        Args:
            yRx (yRxData | numpy.ndarray | list): Received signal to decode.
            chnl_info_list (list[ChanneModel_Rx]): Initial fitted channels.
            generator (list[list[int]] | None): Optional convolutional code
                generator matrix used for extended hypothesis decoding.

        Raises:
            TypeError: If `yRx` is not a supported signal container.
        """

        self.id = DynamicDecoder.next_id
        DynamicDecoder.next_id += 1

        logger.log(f"[DynamicDecoder#{self.id}] Initialized")

        self.chnl_pred_list = []
        self.signal_processor = SignalProcessor()
        self.stationary_processor = StationaryProcessor(n_preamble=chnl_info_list[0].n_preamble)
        if generator is not None:
            self.generator = generator
            
        self.__temp_states = deque(maxlen=self.MAX_TEMP_STATES)
        self.__perm_states = {}
        self._status_active = False
        self._last_decode_progress = -1
        self._last_new_signal_progress = -1
        self._stdout_is_tty = bool(getattr(sys.stdout, "isatty", lambda: False)())

        if isinstance(yRx, list):
            len_yRx = len(yRx)
            self.yRx = np.array(yRx)
        elif isinstance(yRx, yRxData):
            len_yRx = len(yRx.yRxData)
            self.yRx = yRx.yRxData
        elif isinstance(yRx, np.ndarray):
            len_yRx = len(yRx)
            self.yRx = yRx
        else:
            raise TypeError("yRx must be a list, yRxData instance, or numpy array")

        for item in chnl_info_list:
            self.chnl_pred_list.append(ChannelPredictionItem(item, 0, len_yRx))

    def _log(self, message, level="INFO"):
        """Write a decoder-scoped message to the shared logger."""
        logger.log(f"[DynamicDecoder#{self.id}] {message}", level=level)

    def _print(self, message):
        """Print a console message without corrupting the progress line."""
        if self._status_active:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._status_active = False
        print(f"[DynamicDecoder#{self.id}] {message}")

    @staticmethod
    def _progress_bar(current, total, width=24):
        """Render a compact ASCII progress bar."""
        total = max(total, 1)
        current = min(max(current, 0), total)
        filled = int(width * current / total)
        return "#" * filled + "-" * (width - filled)

    def _render_status(self, message):
        """Render a single-line live status message when stdout is interactive."""
        if not self._stdout_is_tty:
            return
        columns = max(40, shutil.get_terminal_size((80, 20)).columns)
        safe_message = message[: max(1, columns - 1)]
        sys.stdout.write("\r" + safe_message.ljust(max(1, columns - 1)))
        sys.stdout.flush()
        self._status_active = True

    def _clear_status(self):
        """Clear the current live status line."""
        if self._status_active and self._stdout_is_tty:
            columns = max(40, shutil.get_terminal_size((80, 20)).columns)
            sys.stdout.write("\r" + (" " * max(1, columns - 1)) + "\r")
            sys.stdout.flush()
        self._status_active = False

    def _update_decode_progress(self, t, signal_num, channel_id=None, err=None, probability=None, abnormal=None):
        """Refresh the main decoding progress display."""
        total = max(len(self.yRx) - 1, 1)
        progress = int((min(t, total) / total) * 100)
        if progress == self._last_decode_progress and t != total:
            return
        self._last_decode_progress = progress
        bar = self._progress_bar(t, total)
        details = f"signals={signal_num} t={t}/{total}"
        self._render_status(
            f"[DynamicDecoder#{self.id}] Decoding [{bar}] {progress:3d}% | {details}"
        )

    def _update_new_signal_progress(self, current, total):
        """Refresh the progress display for new-signal estimation."""
        total = max(int(total), 1)
        progress = int((min(current, total) / total) * 100)
        if progress == self._last_new_signal_progress and current != total:
            return
        self._last_new_signal_progress = progress
        bar = self._progress_bar(current, total)
        self._render_status(
            f"[DynamicDecoder#{self.id}] Estimating new signal [{bar}] {progress:3d}% | combinations={current}/{total}"
        )
            
    def save_state(self, time_ptr, permanent=False, tag=None) -> str:
        """Snapshot the current decoder state for later recovery.

        Args:
            time_ptr (int): Current decode time pointer.
            permanent (bool): Whether to save the state permanently.
            tag (str | None): Optional custom state identifier.

        Returns:
            str: Identifier of the saved state.
        """
        state_id = tag or f"state_{uuid.uuid4().hex[:16]}"
        state_snapshot = {
            "id": state_id,
            "is_permanent": permanent,
            "time_ptr": time_ptr,
            "chnl_pred_states": copy.deepcopy(self.chnl_pred_list)
        }
        
        self._log(f"Saving {'permanent' if permanent else 'temporary'} state {state_id} at t={time_ptr}")
        for chnl in state_snapshot["chnl_pred_states"]:
            self._log(f"State {state_id}: channel={chnl.id}, bits={chnl.bits}, pred_end={chnl.pred_end}")
        
        if permanent:
            self.__perm_states[state_id] = state_snapshot
            self._log(f"Saved permanent state {state_id} with {len(self.chnl_pred_list)} channels")
        else:
            self.__temp_states.append(state_snapshot)
            self._log(f"Saved temporary state {state_id} with {len(self.chnl_pred_list)} channels")
        return state_id
    
    def restore_state(self, state_id: str):
        """Restore a previously saved temporary or permanent state.

        Args:
            state_id (str): Identifier of the saved state.

        Returns:
            int | None: Restored decode time pointer, or None if the state does
            not exist.
        """
        state = next((s for s in self.__temp_states if s["id"] == state_id), None)
        if not state:
            state = self.__perm_states.get(state_id)
        if not state:
            warnings.warn(f"State {state_id} not found in saved states")
            self._log(f"State {state_id} was not found", level="WARNING")
            return None

        self.chnl_pred_list = copy.deepcopy(state["chnl_pred_states"])
        
        self._log(f"Restoring state {state_id} at t={state['time_ptr']}")
        for chnl in self.chnl_pred_list:
            self._log(f"Restored state {state_id}: channel={chnl.id}, bits={chnl.bits}, pred_end={chnl.pred_end}")
        
        self._log(f"Restored state {state_id} with {len(self.chnl_pred_list)} channels")
        
        return state["time_ptr"]
    
    def delete_state(self, state_id: str):
        """Delete a saved decoder snapshot."""
        for i, state in enumerate(self.__temp_states):
            if state["id"] == state_id:
                new_temp_states = deque(maxlen=self.MAX_TEMP_STATES)
                for s in self.__temp_states:
                    if s["id"] != state_id:
                        new_temp_states.append(s)
                self.__temp_states = new_temp_states
                self._log(f"Deleted temporary state {state_id}")
                return
        
        if state_id in self.__perm_states:
            del self.__perm_states[state_id]
            self._log(f"Deleted permanent state {state_id}")
            return
        
        self._log(f"State {state_id} was not found for deletion", level="WARNING")

    def _estimate_bits(self, chnl: ChannelPredictionItem, need_guess_bits_chnl: dict):
        """Estimate the next bit by testing combinations of overlapping channels."""
        keys = list(need_guess_bits_chnl.keys())
        value_lists = [need_guess_bits_chnl[key] for key in keys]

        all_chnl_waveform = np.zeros_like(self.yRx, dtype=np.float64)

        for chnl_item in self.chnl_pred_list:
            all_chnl_waveform += chnl_item.pred

        best_mse = float('inf')
        best_bit = None

        for combination in product(*value_lists):
            chnl_bits_waveform = all_chnl_waveform.copy()
            for i in range(len(keys)):
                _tmp_chnl = keys[i]
                _tmp_bits_list = combination[i]

                for index, _bit in enumerate(_tmp_bits_list):
                    if _bit not in [0, 1]:
                        raise ValueError("Combination elements must be 0 or 1")
                    _bit_wave = _tmp_chnl.generate_bit_waveform(_bit)
                    _chnl_pred_end = _tmp_chnl.pred_end + index * _tmp_chnl.interval
                    end_pos = _chnl_pred_end + len(_bit_wave)
                    
                    if _chnl_pred_end < 0 or _chnl_pred_end >= len(chnl_bits_waveform) or end_pos > len(chnl_bits_waveform):
                        return -1,-1
                    chnl_bits_waveform[_chnl_pred_end: end_pos] += _bit_wave

            for _bit in [0, 1]:
                _bit_wave = chnl.generate_bit_waveform(_bit)
                _chnl_pred_end = chnl.pred_end + len(_bit_wave)
                _chnl_mse_end = chnl.pred_end + chnl.interval
                
                if chnl.pred_end < 0 or _chnl_pred_end > len(chnl_bits_waveform):
                    return -1, -1
                _tmp_chnl_wave_form = chnl_bits_waveform[chnl.pred_end:_chnl_pred_end] + _bit_wave
                
                y_true_segment = self.yRx[chnl.pred_end:_chnl_mse_end]
                y_pred_segment = _tmp_chnl_wave_form[:chnl.interval]
                
                if len(y_true_segment)!= len(y_pred_segment):
                    self._print(
                        "Array length mismatch detected "
                        f"(channel={chnl.id}, bit={_bit}, pred_end={chnl.pred_end}, "
                        f"mse_end={_chnl_mse_end}, bit_wave_len={len(_bit_wave)}, "
                        f"y_true_len={len(y_true_segment)}, y_pred_len={len(y_pred_segment)}, "
                        f"interval={chnl.interval})"
                    )
                    assert(len(y_true_segment) == len(y_pred_segment))
                
                mse = mean_squared_error(y_true_segment, y_pred_segment)
                if mse < best_mse:
                    best_mse = mse
                    best_bit = _bit

        return best_bit, best_mse
    
    def _detect_new_signal(self, abnorml_t, state_id=None, extend_decode_num=Settings.DEFAULT_EXTENDED_BITS_NUM):
        """Try to explain abnormal residuals by fitting an additional channel.

        Args:
            abnorml_t (int): Residual anomaly position.
            state_id (str | None): Saved state to associate with the new
                channel.
            extend_decode_num (int): Number of extra bits used when extending
                channel hypotheses.

        Returns:
            bool: True if a new channel is detected successfully, otherwise
            False.
        """
        self._last_new_signal_progress = -1
        chnl_extended_bits_list = []
        all_chnl_waveform = np.zeros_like(self.yRx, dtype=np.float64)

        for index in range(len(self.chnl_pred_list)):
            all_chnl_waveform += self.chnl_pred_list[index].pred
            _bits_list = hypothesis_extended_viterbi_decode(
                self.chnl_pred_list[index].bits,
                self.generator,
                extend_decode_num=extend_decode_num,
                premble_num=self.chnl_pred_list[index].n_preamble
            )
            chnl_extended_bits_list.append(_bits_list)
            self._log(f"Extended bit hypotheses: {len(_bits_list)} candidates")
        
        logger.save_temp_file(all_chnl_waveform)
        
        total_combinations = np.prod([len(bits) for bits in chnl_extended_bits_list])
        if total_combinations <= Settings.PROCESSING_MULTI_PROCESSING_NUM * 2:
            return self._detect_new_signal_single_process(abnorml_t, state_id, chnl_extended_bits_list)
        
        return self._detect_new_signal_multi_process(abnorml_t, state_id, chnl_extended_bits_list)

    def _detect_new_signal_single_process(self, abnorml_t, state_id, chnl_extended_bits_list):
        """Evaluate new-signal candidates in the current process."""
        best_mse = float('inf')
        best_chnl = None
        best_combination = None
        all_combinations = list(product(*chnl_extended_bits_list))
        total_combinations = len(all_combinations)

        for index, combination in enumerate(all_combinations, start=1):
            result = self._evaluate_combination(combination, abnorml_t)
            if result and result[0] < best_mse:
                best_mse, best_chnl, best_combination = result
            self._update_new_signal_progress(index, total_combinations)

        return self._process_detection_result(best_mse, best_chnl, best_combination, abnorml_t, state_id)

    def _detect_new_signal_multi_process(self, abnorml_t, state_id, chnl_extended_bits_list):
        """Evaluate new-signal candidates with multiprocessing."""
        shared_data = {
            'yRx': self.yRx,
            'abnorml_t': abnorml_t,
            'chnl_pred_list': self.chnl_pred_list,
            'signal_processor': self.signal_processor,
            'stationary_processor': self.stationary_processor
        }
        
        all_combinations = list(product(*chnl_extended_bits_list))
        
        chunk_size = max(1, len(all_combinations) // (Settings.PROCESSING_MULTI_PROCESSING_NUM * 4))
        chunks = [all_combinations[i:i + chunk_size] 
                for i in range(0, len(all_combinations), chunk_size)]
        
        best_mse = float('inf')
        best_chnl = None
        best_combination = None
        
        with mp.Pool(processes=min(Settings.PROCESSING_MULTI_PROCESSING_NUM, mp.cpu_count())) as pool:
            tasks = [(chunk, shared_data) for chunk in chunks]
            
            for index, chunk_result in enumerate(pool.imap_unordered(self._process_chunk, tasks), start=1):
                if chunk_result and chunk_result[0] < best_mse:
                    best_mse, best_chnl, best_combination = chunk_result
                self._update_new_signal_progress(index, len(chunks))
        
        return self._process_detection_result(best_mse, best_chnl, best_combination, abnorml_t, state_id)

    def _process_chunk(self, args):
        """Evaluate one chunk of candidate combinations in a worker task."""
        chunk, shared_data = args
        chunk_best_mse = float('inf')
        chunk_best_chnl = None
        chunk_best_combination = None
        
        for combination in chunk:
            result = self._evaluate_combination_worker(combination, shared_data)
            if result and result[0] < chunk_best_mse:
                chunk_best_mse, chunk_best_chnl, chunk_best_combination = result
        
        if chunk_best_mse < float('inf'):
            return (chunk_best_mse, chunk_best_chnl, chunk_best_combination)
        return None

    def _evaluate_combination(self, combination, abnorml_t):
        """Score one channel-extension combination on the residual waveform."""
        try:
            _chnl_bits_waveform = np.zeros_like(self.yRx, dtype=np.float64)
            
            for i in range(len(combination)):
                _tmp_chnl = self.chnl_pred_list[i]
                _tmp_bits_list = combination[i]
                _chnl_bits_waveform += _tmp_chnl.generate_waveform_from_bits(_tmp_bits_list)
            
            _signal_diff = self.yRx[abnorml_t:] - _chnl_bits_waveform[abnorml_t:]
            _signal_diff_filtered = self.signal_processor.kalman_filter(_signal_diff)
            _signal_diff_filtered = self.signal_processor.adaptive_threshold_filter(_signal_diff_filtered)
            _signal_diff_filtered = self.signal_processor.retain_peak_points_filter(
                peak_points_num=self.stationary_processor.n_preamble + Settings.PEAK_POINT_EXCE_CUT,
                yrx_data=_signal_diff_filtered
            )
            
            tmp_chnl_info, tmp_mse = self.stationary_processor.estimate_channel(
                _signal_diff, _signal_diff_filtered, interval_dev=0, start_pos_dev=0
            )
            
            if tmp_mse < float('inf'):
                return (tmp_mse, tmp_chnl_info, combination)
                
        except Exception as e:
            self._log(f"Combination evaluation failed: {e}", level="WARNING")
        
        return None

    def _evaluate_combination_worker(self, combination, shared_data):
        """Worker-safe variant of combination evaluation for multiprocessing."""
        try:
            yRx = shared_data['yRx']
            abnorml_t = shared_data['abnorml_t']
            chnl_pred_list = shared_data['chnl_pred_list']
            signal_processor = shared_data['signal_processor']
            stationary_processor = shared_data['stationary_processor']
            
            _chnl_bits_waveform = np.zeros_like(yRx, dtype=np.float64)
            
            for i in range(len(combination)):
                _tmp_chnl = chnl_pred_list[i]
                _tmp_bits_list = combination[i]
                _chnl_bits_waveform += _tmp_chnl.generate_waveform_from_bits(_tmp_bits_list)
            
            _signal_diff = yRx[abnorml_t:] - _chnl_bits_waveform[abnorml_t:]
            _signal_diff_filtered = signal_processor.kalman_filter(_signal_diff)
            _signal_diff_filtered = signal_processor.adaptive_threshold_filter(_signal_diff_filtered)
            _signal_diff_filtered = signal_processor.retain_peak_points_filter(
                peak_points_num=stationary_processor.n_preamble + Settings.PEAK_POINT_EXCE_CUT,
                yrx_data=_signal_diff_filtered
            )
            
            tmp_chnl_info, tmp_mse = stationary_processor.estimate_channel(
                _signal_diff, _signal_diff_filtered, interval_dev=0, start_pos_dev=0
            )
            
            if tmp_mse < float('inf'):
                return (tmp_mse, tmp_chnl_info, combination)
                
        except Exception as e:
            pass
        
        return None

    def _process_detection_result(self, best_mse, best_chnl, best_combination, abnorml_t, state_id):
        """Commit the best newly detected channel to the decoder state."""
        self._clear_status()
        if best_mse == float('inf') or best_chnl is None or best_combination is None:
            self._log(f"No valid new channel detected at t={abnorml_t}", level="WARNING")
            self._print("New signal detection failed")
            return False
        
        fig = best_chnl.visualize()
        logger.save_temp_file(best_chnl.generate_preamble())
        logger.save_temp_file(best_chnl.yRx)
        logger.log_image(fig, f"DynamicDecoder[{self.id}]: New Channel Detected", 
                        img_name=f"decoder[{self.id}]_best_chnl")

        new_chnl = ChannelPredictionItem(best_chnl, abnorml_t + best_chnl.start_pos, len(self.yRx))
        
        self._log(f"New signal detected at t={abnorml_t + best_chnl.start_pos} with mse={best_mse:.6f}")
        self._print(f"New signal detected at t={abnorml_t + best_chnl.start_pos} with mse={best_mse:.6f}")
        
        if state_id is not None:
            new_chnl.prev_state_id = state_id
            
        self.chnl_pred_list.append(new_chnl)
        
        return True
    
    def _estimate_bit_in_chnl_list(self, chnl_item: ChannelPredictionItem, _chnl_end: int):
        """Estimate the next bit for one channel while accounting for overlap."""
        need_guess_bits_chnl = {}
        for other_chnl_item in self.chnl_pred_list:

            if other_chnl_item == chnl_item:
                continue
            k = max(0, (
                        _chnl_end - other_chnl_item.pred_end + other_chnl_item.interval - 1) // other_chnl_item.interval)
            if k == 0:
                continue
            possible_combinations = list(product([0, 1], repeat=k))
            need_guess_bits_chnl[other_chnl_item] = possible_combinations

        _bit, _err = self._estimate_bits(chnl_item, need_guess_bits_chnl)
        return _bit, _err

    def decode(self, MAX_SIGNAL_NUM = None, visualize=False):
        """Run the full dynamic decoding loop.

        Args:
            MAX_SIGNAL_NUM (int | None): Optional cap on the number of signals
                that may be tracked simultaneously.
            visualize (bool): Whether to display intermediate or final plots.

        Returns:
            list[str]: Decoded bit strings for all tracked channels.
        """
        SIGNAL_NUM = len(self.chnl_pred_list)
        self._log("Starting decode loop")
        
        IN_RESET_MODE = False
        NEW_SIGNAL_DETECT_VALUE = 0
        NEW_SIGNAL_STATE_ID = None
        DETECT_NEW_SIGNAL_FLAG =False
        NEED_TO_DETECT_NEW_SIGNAL = True
        
        t = 0
        while t < len(self.yRx):
            self._update_decode_progress(t, SIGNAL_NUM)
            reset_state_id = None
            NEW_SIGNAL_reset_state_id = None
            assert(SIGNAL_NUM == len(self.chnl_pred_list)), f"Signal number mismatch: SIGNAL_NUM={SIGNAL_NUM}, len(chnl_pred_list)={len(self.chnl_pred_list)}"
            
            if NEED_TO_DETECT_NEW_SIGNAL:
                ALLOW_NEW_SIGNAL_DETECT = True
            else:
                ALLOW_NEW_SIGNAL_DETECT = False
                
            for chnl_item in self.chnl_pred_list:
                if len(chnl_item.bits) <= len(chnl_item.preamble_bits) + len(Settings.CHECK_CODE):
                    ALLOW_NEW_SIGNAL_DETECT = False
                else:
                    chnl_item.prev_state_id = None
                    
            for chnl_item in self.chnl_pred_list:
                _chnl_end = chnl_item.pred_end + chnl_item.interval

                if _chnl_end > len(self.yRx):
                    self._clear_status()
                    self._print("Decode finished: all channels have been processed")
                    self._log("Decode finished: all channels have been processed")
                    self.draw_signal(visualize=visualize)
                    return [item.bits for item in self.chnl_pred_list]

                if _chnl_end <= t:
                    _bit, _err = self._estimate_bit_in_chnl_list(chnl_item, _chnl_end)
                    
                    if _bit == -1 and _err == -1:
                        self._clear_status()
                        self._print("Decode ended")
                        self._log("Decode ended")
                        self.draw_signal(visualize=visualize)
                        return [item.bits for item in self.chnl_pred_list]
                    
                    is_abnormal, _pb_value = chnl_item.detect_err(_err)
                    self._log(
                        f"t={t}, channel={chnl_item.id}, bits={chnl_item.bits}, "
                        f"bit={_bit}, err={_err}, probability={_pb_value}"
                    )
                    self._update_decode_progress(
                        t,
                        SIGNAL_NUM,
                        channel_id=chnl_item.id,
                        err=_err,
                        probability=_pb_value,
                        abnormal=is_abnormal,
                    )
                    
                    if not DETECT_NEW_SIGNAL_FLAG and not IN_RESET_MODE:
                        if is_abnormal:
                            if NEW_SIGNAL_DETECT_VALUE == 0:
                                NEW_SIGNAL_STATE_ID = self.save_state(t, permanent=False)
                            if _pb_value >= Settings.SURE_ERR_PROB_THRESHOLD:
                                NEW_SIGNAL_DETECT_VALUE += 1
                            else:
                                NEW_SIGNAL_DETECT_VALUE += Settings.NEW_SIGNAL_DETECT_THRESHOLD
                                
                            self._log(
                                f"Abnormal condition detected for channel={chnl_item.id} "
                                f"at t={t}; detector_value={NEW_SIGNAL_DETECT_VALUE}"
                            )
                        else:
                            NEW_SIGNAL_DETECT_VALUE = max(0, NEW_SIGNAL_DETECT_VALUE - Settings.NEW_SIGNAL_DETECT_RECOVERY)
                        
                        if NEW_SIGNAL_DETECT_VALUE >= Settings.NEW_SIGNAL_DETECT_THRESHOLD:
                            DETECT_NEW_SIGNAL_FLAG = True
                            NEW_SIGNAL_DETECT_VALUE = 0
                            NEW_SIGNAL_reset_state_id = NEW_SIGNAL_STATE_ID
                            break
                        
                    
                    if IN_RESET_MODE:
                        chnl_item.update_bit(_bit)
                        chnl_item.add_err(_err)
                        
                    elif (ALLOW_NEW_SIGNAL_DETECT and DETECT_NEW_SIGNAL_FLAG and chnl_item.prev_state_id is None):
                        self._print("Triggering new-signal detection")
                        state_id = NEW_SIGNAL_STATE_ID
                        SIGNAL_NUM += 1
                        if MAX_SIGNAL_NUM is not None and SIGNAL_NUM > MAX_SIGNAL_NUM:
                            self._log(
                                f"Reached maximum signal limit at t={t} "
                                f"(signal_num={SIGNAL_NUM}, max_signal_num={MAX_SIGNAL_NUM})",
                                level="WARNING",
                            )
                            self._print(
                                f"Reached maximum signal limit at t={t} "
                                f"(signal_num={SIGNAL_NUM}, max_signal_num={MAX_SIGNAL_NUM})"
                            )
                            NEED_TO_DETECT_NEW_SIGNAL = False
                            SIGNAL_NUM -= 1
                        else:
                            self._log(
                                f"New-signal trigger at t={t}, channel={chnl_item.id}, "
                                f"bits={chnl_item.bits}, bit={_bit}, err={_err}, probability={_pb_value}"
                            )
                            self._print(
                                f"New-signal trigger at t={t}, bits={chnl_item.bits}, "
                                f"pred_end={chnl_item.pred_end}, err={_err:.6f}"
                            )
                            self.draw_signal(visualize=visualize)
                            _ = self._detect_new_signal(chnl_item.pred_end, state_id=state_id)
                            
                            _bit, _err = self._estimate_bit_in_chnl_list(chnl_item, _chnl_end)
                            chnl_item.update_bit(_bit)
                            chnl_item.add_err(_err)
                            
                            if not _:
                                self._print("New signal detection failed")
                                NEED_TO_DETECT_NEW_SIGNAL = False
                                SIGNAL_NUM -= 1
                            else:
                                for chnl_item in self.chnl_pred_list:
                                    chnl_item.reset_abnormal_detector()
                        
                    else:
                        if not is_abnormal:
                            chnl_item.add_err(_err)
                        chnl_item.update_bit(_bit)
                        
                    _com_preamble_code = chnl_item.preamble_bits + Settings.CHECK_CODE
                    if len(chnl_item.bits) == len(_com_preamble_code):
                        self._log(f"Channel {chnl_item.id} completed preamble decoding")
                        assert(len(chnl_item.bits) != 0)
                        mismatches = sum(1 for i in range(len(_com_preamble_code)) if chnl_item.bits[i] != _com_preamble_code[i])
                        error_rate = mismatches / len(_com_preamble_code)
                        self._print(f"Channel {chnl_item.id} preamble error rate: {error_rate:.2%}")
                        self._log(
                            f"Channel {chnl_item.id} preamble error rate={error_rate:.2%}; "
                            f"threshold={Settings.ALLOW_BITS_ERRORPEC_IN_DETECT[SIGNAL_NUM - 1]:.2%}"
                        )
                        if error_rate > Settings.ALLOW_BITS_ERRORPEC_IN_DETECT[SIGNAL_NUM - 1]:
                            self._log(
                                f"Channel {chnl_item.id} preamble error rate is too high: {error_rate:.2%}",
                                level="WARNING",
                            )
                            self._print(f"Warning: channel {chnl_item.id} preamble error rate is too high: {error_rate:.2%}")
                            if chnl_item.prev_state_id is not None:
                                reset_state_id = chnl_item.prev_state_id
                                break
                                
            if reset_state_id is not None:
                tmp = self.restore_state(reset_state_id)
                if tmp is not None:
                    t = tmp
                    SIGNAL_NUM = len(self.chnl_pred_list)
                    self._log(f"Restored state {reset_state_id} at t={t} due to high preamble error rate")
                    self._print(f"Restored state {reset_state_id} at t={t} due to high preamble error rate")
                    IN_RESET_MODE = True
                    DETECT_NEW_SIGNAL_FLAG = False
                else:
                    DETECT_NEW_SIGNAL_FLAG = False
                    IN_RESET_MODE = False
                    t += 1
                    reset_state_id = None
            elif NEW_SIGNAL_reset_state_id is not None:
                tmp = self.restore_state(NEW_SIGNAL_reset_state_id)
                if tmp is not None:
                    t = tmp
                    SIGNAL_NUM = len(self.chnl_pred_list)
                    self._log(f"Restored state {NEW_SIGNAL_reset_state_id} at t={t} for new-signal detection")
                    self._print(f"Restored state {NEW_SIGNAL_reset_state_id} at t={t} for new-signal detection")
                    IN_RESET_MODE = False
                    DETECT_NEW_SIGNAL_FLAG = True
                else:
                    DETECT_NEW_SIGNAL_FLAG = False
                    IN_RESET_MODE = False
                    t += 1
                    NEW_SIGNAL_reset_state_id = None
            else:
                DETECT_NEW_SIGNAL_FLAG = False
                IN_RESET_MODE = False
                t += 1
            

        self._print("Decode ended")
        self._log("Decode ended")
        self.draw_signal(visualize=visualize)
        return [item.bits for item in self.chnl_pred_list]

    def draw_signal(self, visualize=False):
        """Log or display the current aggregate channel reconstruction.

        Args:
            visualize (bool): Whether to display the reconstructed waveform in
                a matplotlib window.
        """
        all_chnl_waveform = np.zeros_like(self.yRx, dtype=np.float64)
        for chnl_item in self.chnl_pred_list:
            all_chnl_waveform += chnl_item.pred
        logger.log(self.yRx, description=f"[DynamicDecoder#{self.id}] signal overview", predict_signal=all_chnl_waveform,
                   img_save_subfolder=f"decoder")
        if visualize:
            fig = Plotter.draw_originPic_predictPic(self.yRx, all_chnl_waveform)
            fig.show()



