"""Core transmit and receive channel models for AdaptMolMAC.

This module defines the main data container for received signals together with
the transmitter-side and receiver-side channel abstractions used throughout the
package.
"""

from ..sim import MCModel, noiseParam
from ..mcutils import logger, BinaryUtils, generate_preamble_bits
from ..viterbi import PATH_DRAW, validate_viterbi_generator, convolutional_encode
from ..config import Settings

from sklearn.metrics import mean_squared_error
from scipy.signal import correlate
import matplotlib.pyplot as plt
import numpy as np
import warnings

PATH_DRAW = False

class yRxData:
    """Store raw and processed received-signal samples.

    The object lazily applies the configured noise model when the processed
    signal is first requested. It is also used as the common data container
    between simulation, preprocessing, and decoding stages.

    Attributes:
        send_bits (list | str): Labels associated with the signal payload.
        is_locked (bool): Whether the raw buffer has been discarded.
        ifLogger (bool): Whether initialization and state changes are logged.
    """

    def __init__(self, data=None, send_bits=None, process_data=None, ifLogger = True):
        """Initialize the signal container.

        Args:
            data (array-like | None): Raw noiseless samples.
            send_bits (list | str | None): Bit labels associated with the
                signal.
            process_data (array-like | None): Preprocessed samples to store
                directly.
            ifLogger (bool): Whether to emit logger messages.

        Raises:
            ValueError: If neither `data` nor `process_data` is provided when a
                non-empty instance is expected.
        """
        self.send_bits = send_bits if send_bits is not None else []
        self.is_locked = False
        self.ifLogger = ifLogger

        if data is None and process_data is None:
            self._data = np.array([], dtype=float)
            self._processed_data = np.array([], dtype=float)
            if self.ifLogger:
                logger.log(f"[yRxData] Initialized empty container with send_bits={self.send_bits}")
            return

        if process_data is not None:
            self._processed_data = np.array(process_data, copy=True)
            self.is_locked = True
            if self.ifLogger:
                logger.log(f"[yRxData] Initialized locked container with send_bits={self.send_bits}")
                # logger.log(self.yRxData, description=f"yRxData:{self.send_bits}")
        else:
            if data is None:
                raise ValueError("Either 'data' or 'process_data' parameter must be provided")
            self._data = np.array(data, copy=True)
            self._processed_data = None
            if self.ifLogger:
                logger.log(f"[yRxData] Initialized with send_bits={self.send_bits}")
                # logger.log(self.yRxData, description=f"yRxData:{self.send_bits}")

    def __getitem__(self, index):
        """Return one processed sample by index."""
        return self._processed_data[index]

    def __add__(self, other):
        """Overlay two unlocked signals into one container.

        Args:
            other (yRxData): Another signal container to merge with.

        Returns:
            yRxData: A new container holding the summed raw signal.

        Raises:
            RuntimeError: If either signal has already been locked.
        """
        if self.is_locked or other.is_locked:
            raise RuntimeError("Cannot add locked yRxData instances")
        
        if self.ifLogger:
            logger.log(f"[yRxData] Merging send_bits={self.send_bits} + {other.send_bits}")
        
        data = np.zeros(max(len(self.raw_data), len(other.raw_data)))
        data[:len(self.raw_data)] += self.raw_data
        data[:len(other.raw_data)] += other.raw_data
        send_bits = self.send_bits + other.send_bits
        return yRxData(data, send_bits=send_bits)

    @property
    def raw_data(self):
        """Return the raw noiseless signal.

        Returns:
            numpy.ndarray: The original raw signal buffer.

        Raises:
            AttributeError: If the container has already been locked.
        """
        if self.is_locked:
            raise AttributeError("Raw data is cleared after locking")
        return self._data

    @property
    def yRxData(self):
        """Return the processed signal representation.

        Returns:
            numpy.ndarray: The processed signal. If only raw data is available,
            noise is injected the first time this property is accessed.
        """
        if self._processed_data is None:
            processed = noiseParam.AddNoise(self._data)
            self._processed_data = np.array(processed, copy=True)
        return self._processed_data

    def lock(self):
        """Freeze the processed signal and release the raw buffer."""
        
        if self.ifLogger:
            logger.log(f"[yRxData] Locking container with send_bits={self.send_bits}")
        
        if not self.is_locked:
            _ = self.yRxData
            del self._data
            self.is_locked = True

    def __repr__(self):
        return repr(self.yRxData)
    
    def visualize(self):
        """Visualize the waveform and its associated transmitted bits.

        Returns:
            matplotlib.figure.Figure: A figure containing the waveform plot and
            bit annotations.
        """
        x_time = list(range(len(self.yRxData)))
        fig, axs = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})

        # Plot the received signal
        axs[0].plot(x_time, self.yRxData, label='Received Signal', color='blue')
        axs[0].set_title('Received Signal Visualization', fontsize=14)
        axs[0].set_xlabel('Time Index', fontsize=12)
        axs[0].set_ylabel('Amplitude', fontsize=12)
        axs[0].legend()
        axs[0].grid(True)

        # Display the send_bits as text
        send_bits_str = ''.join(map(str, self.send_bits))
        send_bits_str = '\n    '.join(self.send_bits)  # Join list elements with indentation
        axs[1].text(0.05, 0.5, f'Sent Bits:\n    {send_bits_str}', 
            fontsize=18, ha='left', va='center', wrap=True, 
            color='black', linespacing=1.5)
        axs[1].axis('off')  # Hide axes for the text display

        return fig

class ChannelModel_Tx:
    """Generate transmit waveforms with optional coding and preamble insertion.

    This class is the main transmitter-side interface exposed by the package.
    It wraps the lower-level simulation backend and keeps track of transmitted
    payloads and encoded sequences.

    Attributes:
        id (int): Instance identifier used in logs.
        viterbi_gen (list[list[int]] | None): Optional convolutional code
            generator matrix.
        premble (str | None): Generated preamble bit string.
        n_premble (int | None): Preamble order.
        transmit_data (list[str]): Original payloads sent by this instance.
        encode_data (list[str]): Encoded payloads produced before simulation.
    """

    next_id = 0
    def __init__(self, interval, tx_offset, amplitude=1.0, n_preamble=None, viterbi_gen=None, ChannelParam = None):
        """Configure a transmitter with timing, amplitude, and coding options.

        Args:
            interval (int): Symbol interval used during transmission.
            tx_offset (int): Transmit offset applied before the waveform starts.
            amplitude (float): Output amplitude scaling factor.
            n_preamble (int | None): Number of preamble groups to prepend.
            viterbi_gen (list[list[int]] | None): Optional convolutional coding
                generator matrix.
            ChannelParam (list[float] | None): Optional custom channel
                parameters for the simulator.
        """
        self.id = ChannelModel_Tx.next_id
        ChannelModel_Tx.next_id += 1

        self._model = MCModel(ChannelParam = ChannelParam)
        self._configure(interval, tx_offset, amplitude)

        self.viterbi_gen = viterbi_gen
        if self.viterbi_gen is not None:
            self.constraint_length = validate_viterbi_generator(self.viterbi_gen)

        if n_preamble is None:
            self.premble = None
            self.n_premble = None
        else:
            self.premble = generate_preamble_bits(n_preamble)
            self.n_premble = n_preamble
        self.transmit_data = []
        self.encode_data = []
        logger.log(
            f"[ChannelModel_Tx#{self.id}] Initialized "
            f"(interval={interval}, tx_offset={tx_offset}, amplitude={amplitude}, "
            f"preamble={self.premble}, viterbi_gen={viterbi_gen})"
        )
    
    @staticmethod
    def set_noise_param(noiseb, noisen, noisep):
        """Update the global simulation noise parameters."""
        noiseParam.set_noise_params(noiseb=noiseb, noisen=noisen, noisep=noisep)

    def _configure(self, interval, tx_offset, amplitude):
        """Apply the basic waveform-generation parameters to the simulator."""
        self._model.setInterval(int(interval))
        self._model.setConsisTxOffset(int(tx_offset))
        self._model.setAmplitude(float(amplitude))

    def transmit(self, data):
        """Encode and transmit a payload bit string.

        Args:
            data (str): Payload bits to transmit.

        Returns:
            yRxData: The simulated received waveform with metadata attached.
        """
        if self.viterbi_gen is not None:
            encode_data = convolutional_encode(data, self.viterbi_gen, self.constraint_length)
        else:
            encode_data = data
        if self.premble is not None:
            encode_data = self.premble + Settings.CHECK_CODE + encode_data
        BinaryUtils.validate_binary_string(encode_data)
        _yRx = self._model.send(encode_data, AddNoise=False)
        logger.log(f"[ChannelModel_Tx#{self.id}] Transmitting data={data} with encoded_data={encode_data}")
        
        self.transmit_data.append(data)
        self.encode_data.append(encode_data)
        
        _yRx = yRxData(_yRx, send_bits=[encode_data])
        return _yRx
        
    @property
    def parameters(self):
        """Return the transmitter timing configuration.

        Returns:
            dict: A dictionary containing interval and offset settings.
        """
        return {
            'interval': self._model.interval,
            'tx_offset': self._model.tx_offset
        }

class ChanneModel_Rx:
    """Represent a fitted receiver-side waveform template.

    The model is estimated from preamble peaks and later used to generate
    waveform predictions for bit-by-bit dynamic decoding.

    Attributes:
        id (int): Instance identifier used in logs.
        interval (int): Estimated symbol interval.
        sig_end (int): Length of a single-bit waveform template.
        key_point1 (list[float]): First waveform key point `[x, y]`.
        key_point2 (list[float]): Second waveform key point `[x, y]`.
        preamble_bits (str): Preamble used when fitting the template.
        start_pos (int): Current waveform start offset.
    """

    next_id = 0
    def __init__(self, interval, peak1_x, yRx, n_preamble, n_peak, sig_end = None):
        """Initialize a receiver template from coarse timing observations.

        Args:
            interval (int): Estimated symbol interval.
            peak1_x (int): Position of the first detected peak.
            yRx (yRxData | numpy.ndarray | list): Received signal samples.
            n_preamble (int): Preamble order used for fitting.
            n_peak (int): Number of peaks expected from the preamble.
            sig_end (int | None): Optional template length override.
        """
        self.id = ChanneModel_Rx.next_id
        ChanneModel_Rx.next_id += 1
        self.n_peak = n_peak
        self.interval = int(interval)
        if sig_end is None:
            self.sig_end = int(self.interval * 3 + 1)
        else:
            self.sig_end = sig_end
        self.yRx = yRx
        
        peak1_y_sum = 0
        for i in range(n_peak):
            peak1_y_sum += yRx[peak1_x + (i*(i+1)//2) * self.interval]

        self.key_point1 = [peak1_x , self.cal_key1_value(peak1_x)]
        self.key_point2 = [self.sig_end, 0]
        if isinstance(yRx, yRxData):
            self.yRx = yRx.yRxData
        else:
            self.yRx = yRx
        self.n_preamble = n_preamble
        self.preamble_bits = generate_preamble_bits(n_preamble)
        self.start_pos = 0
        logger.log(
            f"[ChannelModel_Rx#{self.id}] Initialized "
            f"(interval={interval}, peak1_x={peak1_x}, sig_end={sig_end}, n_preamble={n_preamble})"
        )
        
    def fresh_key1_value(self):
        """Refresh the first key point amplitude from the current signal."""
        # self.key_point1[1] = self.cal_key1_value(self.key_point1[0])
        self.key_point1[1] = self.yRx[self.key_point1[0]]
        
    def cal_key1_value(self, x):
        """Estimate the first key point amplitude by averaging preamble peaks."""
        peak1_y_sum = 0
        for i in range(self.n_peak):
            peak1_y_sum += self.yRx[x + (i*(i+1)//2) * self.interval]
        return peak1_y_sum / self.n_peak

    def visualize_bit(self, bit_value=1):
        """Plot the template waveform generated for one bit value."""
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(self.sig_end), self.generate_bit_waveform(bit_value), label=f'Bit{bit_value} Waveform')
        ax.scatter([self.key_point1[0], self.key_point2[0]],
                   [self.key_point1[1], self.key_point2[1]],
                   color='red', zorder=5)
        ax.set_title('Bit1 Signal Structure')
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')
        ax.legend()
        ax.grid(True)
        return fig

    def _validate_keypoints(self):
        """Validate the relative ordering of the template key points."""
        if self.key_point1[0] >= self.key_point2[0]:
            raise ValueError("Key point 1 must appear before key point 2.")
        if self.key_point2[0] > self.sig_end:
            raise ValueError("Signal duration exceeds the configured range.")
        
    # def train_keypoint(self, interval_dev=0, start_pos_dev=0, num_segments=10, min_interval_size=1):
    #     best_mse = float('inf')
    #     optimal_kp2 = None
    #     init_key1_x = self.key_point1[0]
    #     logger.log(f"Rx[{self.id}] initial key point 1: {self.key_point1}")
        
    #     print("interval:",self.interval)

    #     interval_range = list(range(self.interval - interval_dev, self.interval + interval_dev + 1))
    #     start_pos_range = list(range(start_pos_dev + 1))
    #     best_interval = self.interval
    #     best_start_pos = 0
    #     logger.log(f"Rx[{self.id}] Interval range: {interval_range}, Start position range: {start_pos_range}")

    #     preamble_bits = self.preamble_bits
    #     for interval in interval_range:
    #         self.interval = interval
    #         for start_pos in start_pos_range:
    #             self.key_point1[0] = init_key1_x - start_pos
    #             x1 = self.key_point1[0]
    #             y1 = self.key_point1[1]
    #             wave_length = int(len(preamble_bits) * self.interval)

    #             if wave_length + start_pos > len(self.yRx):
    #                 continue

    #             y_segment = np.array(self.yRx[start_pos:start_pos + wave_length])
                
    #             def calculate_mse(x2):
    #                 A_total, b_total = self.cal_matrix_A_b(preamble_bits, x1, y1, x2, interval, self.sig_end)
    #                 A_total = A_total[:wave_length]
    #                 b_total = b_total[:wave_length]
                    
    #                 if A_total.shape[0] != wave_length:
    #                     return float('inf'), 0
                    
    #                 denominator = np.dot(A_total, A_total)
    #                 if denominator < 1e-10:
    #                     y2_opt = 0
    #                 else:
    #                     numerator = np.dot(A_total, y_segment - b_total)
    #                     y2_opt = numerator / denominator
    #                 y2_opt = np.clip(y2_opt, 0, y1)
    #                 wave_generated = A_total * y2_opt + b_total
    #                 return mean_squared_error(y_segment, wave_generated), y2_opt

                
    #             low = x1
    #             high = self.sig_end
                
    #             while high - low > min_interval_size:
    #                 actual_segments = min(num_segments, high - low)
                    
    #                 segment_points = sorted(set(np.linspace(low, high, actual_segments + 1, dtype=int)))
                    
    #                 min_mse = float('inf')
    #                 best_segment = 0
                    
    #                 for i, x2 in enumerate(segment_points):
    #                     current_mse, _ = calculate_mse(x2)
    #                     if current_mse < min_mse:
    #                         min_mse = current_mse
    #                         best_segment = i
                    
    #                 new_low_idx = max(0, best_segment - 1)
    #                 new_high_idx = min(len(segment_points)-1, best_segment + 1)
    #                 low = segment_points[new_low_idx]
    #                 high = segment_points[new_high_idx]
                
    #             search_points = range(low, high + 1)
    #             for x2 in search_points:
    #                 current_mse, y2_opt = calculate_mse(x2)
                    
    #                 if current_mse < best_mse:
    #                     best_mse = current_mse
    #                     optimal_kp2 = [x2, y2_opt]
    #                     best_interval = interval
    #                     best_start_pos = start_pos

    #             logger.log(f"Rx[{self.id}] Interval: {interval}, Start Position: {start_pos}, best_mse = {best_mse}")
    #     self.key_point2 = optimal_kp2
    #     self.key_point1[0] = init_key1_x - best_start_pos
    #     self.interval = best_interval
    #     self.start_pos = best_start_pos

    #     if self.key_point2 is None:
    #         logger.log(f"Rx[{self.id}] Unable to find a valid key point 2", level='WARNING')
    #         warnings.warn("Unable to find a valid key point 2")

    #     logger.log(f"Rx[{self.id}] Optimized key points: {self.key_point1}, {self.key_point2}, Best MSE: {best_mse}, Interval: {best_interval}, Start Position: {best_start_pos}")

    #     return best_mse
    
    def train_keypoint(self, interval_dev=0, start_pos_dev=0):
        """Search for the best waveform key points that match the preamble.

        Args:
            interval_dev (int): Search radius applied to the interval.
            start_pos_dev (int): Search radius applied to the start position.

        Returns:
            float: The best mean squared error found during fitting.

        Raises:
            Warning: Emits a warning when no valid second key point is found.
        """
        best_mse = float('inf')
        optimal_kp2 = None
        init_key1_x = self.key_point1[0]
        logger.log(f"[ChannelModel_Rx#{self.id}] Initial key point 1: {self.key_point1}")

        interval_range = list(range(self.interval - interval_dev, self.interval + interval_dev + 1))
        start_pos_range = list(range(start_pos_dev + 1))
        best_interval = self.interval
        best_start_pos = 0
        logger.log(
            f"[ChannelModel_Rx#{self.id}] Search ranges: "
            f"intervals={interval_range}, start_positions={start_pos_range}"
        )

        preamble_bits = self.preamble_bits

        for interval in interval_range:
            self.interval = interval
            for start_pos in start_pos_range:
                self.key_point1[0] = init_key1_x - start_pos
                self.fresh_key1_value()
                x1 = self.key_point1[0]
                y1 = self.key_point1[1]
                x_range = range(x1, self.sig_end + 1)
                wave_length = int(len(preamble_bits) * self.interval)

                if wave_length + start_pos > len(self.yRx):
                    continue

                for x2 in x_range:
                    
                    A_total, b_total = self.cal_matrix_A_b(preamble_bits, x1, y1, x2, interval, self.sig_end)
                    A_total = A_total[:wave_length]
                    b_total = b_total[:wave_length]
                    y_segment = np.array(self.yRx[start_pos:start_pos + wave_length])
                    
                    if(wave_length + start_pos > len(self.yRx)):
                        continue
                        raise ValueError("The generated waveform length exceeds the received signal range")
                    if A_total.shape[0] != wave_length:
                        continue
                        raise ValueError("Generated waveform length is insufficient")

                    denominator = np.dot(A_total, A_total)
                    if denominator < 1e-10:
                        y2_opt = 0
                    else:
                        numerator = np.dot(A_total, y_segment - b_total)
                        y2_opt = numerator / denominator
                    y2_opt = np.clip(y2_opt, 0, y1)
                    wave_generated = A_total * y2_opt + b_total
                    current_mse = mean_squared_error(y_segment, wave_generated)

                    
                    if current_mse <= best_mse:
                        best_mse = current_mse
                        optimal_kp2 = [x2, y2_opt]
                        best_interval = interval
                        best_start_pos = start_pos

                logger.log(
                    f"[ChannelModel_Rx#{self.id}] interval={interval}, "
                    f"start_pos={start_pos}, best_mse={best_mse}"
                )

        self.key_point2 = optimal_kp2
        self.key_point1[0] = init_key1_x - best_start_pos
        self.interval = best_interval
        self.start_pos += best_start_pos

        if self.key_point2 is None:
            logger.log(f"[ChannelModel_Rx#{self.id}] Unable to find a valid key point 2", level='WARNING')
            warnings.warn("Unable to find a valid key point 2")

        logger.log(
            f"[ChannelModel_Rx#{self.id}] Optimized key points: "
            f"key_point1={self.key_point1}, key_point2={self.key_point2}, "
            f"best_mse={best_mse}, interval={best_interval}, start_pos={best_start_pos}"
        )
        logger.log(
            f"[ChannelModel_Rx#{self.id}] Training complete | "
            f"key_point1={self.key_point1}, key_point2={self.key_point2}, "
            f"mse={best_mse:.4f}, interval={best_interval}, start_pos={best_start_pos}"
        )
        
        self.reset_start_pos()
        self.fresh_key1_value()

        return best_mse
    
    def fine_tune_params(self, delta_start=5, delta_interval=5):
        """Refine start position and interval around the coarse estimate.

        Args:
            delta_start (int): Search radius for the start position.
            delta_interval (int): Search radius for the interval.

        Returns:
            float: The best mean squared error after fine tuning.
        """
        start_pos0 = self.start_pos
        interval0 = self.interval
        preamble = self.preamble_bits

        best_mse = float('inf')
        best_start = start_pos0
        best_interval = interval0

        logger.log(
            f"[ChannelModel_Rx#{self.id}] Fine-tuning parameters: "
            f"start_pos0={start_pos0}, interval0={interval0}, "
            f"delta_start={delta_start}, delta_interval={delta_interval}"
        )

        templates = {}
        for d_offset in range(-delta_interval, delta_interval + 1):
            d = interval0 + d_offset
            self.interval = d
            templates[d] = self.generate_preamble()
            logger.log(f"[ChannelModel_Rx#{self.id}] Generated template for interval={d}")

        self.interval = interval0

        for d, template in templates.items():
            wave_length = len(preamble) * d

            corr = correlate(self.yRx, template[:wave_length], mode='valid')

            start_min = max(0, start_pos0 - delta_start)
            start_max = min(len(corr), start_pos0 + delta_start + 1)

            max_idx = np.argmax(corr[start_min:start_max]) + start_min
            start_pos_candidate = max_idx

            y_segment = self.yRx[start_pos_candidate:start_pos_candidate+wave_length]
            t_segment = template[:wave_length]
            mse = mean_squared_error(y_segment, t_segment)

            logger.log(
                f"[ChannelModel_Rx#{self.id}] interval={d}, "
                f"start_pos_candidate={start_pos_candidate}, mse={mse:.6f}"
            )

            if mse < best_mse:
                best_mse = mse
                best_start = start_pos_candidate
                best_interval = d

        self.key_point1[0] = self.key_point1[0] - best_start
        self.interval = best_interval
        self.start_pos += best_start

        logger.log(
            f"[ChannelModel_Rx#{self.id}] Fine-tuning result: "
            f"best_start={best_start}, best_interval={best_interval}, best_mse={best_mse:.6f}"
        )
        logger.log(
            f"[ChannelModel_Rx#{self.id}] Fine-tuning complete | "
            f"best_start_pos={best_start}, best_interval={best_interval}, "
            f"mse={best_mse:.6f}"
        )
        
        self.reset_start_pos()

        return best_mse


    def generate_bit_waveform(self, bit_value):
        """Generate the waveform template for one bit value.

        Args:
            bit_value (int | str): Bit value, either 0/1 or "0"/"1".

        Returns:
            numpy.ndarray: The generated single-bit waveform template.
        """
        try:
            assert bit_value in (0, 1, '0', '1')
            
            if bit_value == 0 or bit_value == '0':
                return np.zeros(self.sig_end)
            
            if not hasattr(self, 'key_point1') or not hasattr(self, 'key_point2'):
                return np.zeros(self.sig_end)
                
            if self.sig_end <= 0:
                return np.zeros(self.sig_end)
            
            waveform = np.zeros(self.sig_end)
            
            key1_idx = max(0, min(int(self.key_point1[0]), self.sig_end - 1))
            key2_idx = max(0, min(int(self.key_point2[0]), self.sig_end - 1))
            
            if key1_idx >= key2_idx:
                if key1_idx < self.sig_end:
                    waveform[:key1_idx+1] = np.linspace(0, self.key_point1[1], key1_idx+1)
                return waveform
            
            if key1_idx > 0:
                seg1_length = key1_idx + 1
                if seg1_length > 0:
                    seg1_values = np.linspace(0, self.key_point1[1], seg1_length)
                    if len(seg1_values) == min(seg1_length, self.sig_end):
                        waveform[:seg1_length] = seg1_values[:min(seg1_length, self.sig_end)]
            
            seg2_length = key2_idx - key1_idx
            if seg2_length > 0:
                seg2_values = np.linspace(self.key_point1[1], self.key_point2[1], seg2_length)
                target_start = key1_idx
                target_end = key2_idx
                actual_target_length = target_end - target_start
                
                if actual_target_length == len(seg2_values) and target_start < self.sig_end:
                    end_idx = min(target_end, self.sig_end)
                    if target_start < end_idx:
                        waveform[target_start:end_idx] = seg2_values[:end_idx-target_start]
            
            if key2_idx < self.sig_end:
                seg3_length = self.sig_end - key2_idx
                if seg3_length > 0:
                    seg3_values = np.linspace(self.key_point2[1], 0, seg3_length)
                    if len(seg3_values) == seg3_length:
                        waveform[key2_idx:] = seg3_values
            
            return waveform
            
        except Exception as e:
            print(
                f"[ChannelModel_Rx#{self.id}] generate_bit_waveform failed: {e} | "
                f"bit_value={bit_value}, sig_end={self.sig_end}, "
                f"key_point1={getattr(self, 'key_point1', 'None')}, "
                f"key_point2={getattr(self, 'key_point2', 'None')}"
            )
            return np.zeros(self.sig_end)

    def train_keypoint_old(self, interval_dev = 0, start_pos_dev = 0, n_iter=100):
        """Legacy exhaustive key-point search kept for comparison/debugging."""
        best_mse = float('inf')
        optimal_kp2 = None
        init_key1_x = self.key_point1[0]
        logger.log(f"Rx[{self.id}] initial key point 1: {self.key_point1}")

        interval_range = list(range(self.interval - interval_dev, self.interval + interval_dev + 1))
        start_pos_range = list(range(start_pos_dev + 1))
        best_interval = self.interval
        best_start_pos = 0
        logger.log(f"Rx[{self.id}] Interval range: {interval_range}, Start position range: {start_pos_range}")
        
        for interval in interval_range:
            self.interval = interval
            for start_pos in start_pos_range:
                self.key_point1[0] = init_key1_x-start_pos
                x_range = range(self.key_point1[0], self.sig_end + 1)
                y_range = np.linspace(0, self.key_point1[1], n_iter)
                wave_length = int(len(self.preamble_bits)*interval)
                for x in x_range:
                    for y in y_range:
                        self.key_point2 = [x, y]
                        #try:
                        if True:
                            wave_generated = self.generate_preamble()
                            if(wave_length + start_pos > len(self.yRx)):
                                continue
                                raise ValueError("The generated waveform length exceeds the received signal range")
                            if len(wave_generated) < wave_length:
                                continue
                                raise ValueError("Generated waveform length is insufficient")
                            current_mse = mean_squared_error(np.array(self.yRx[start_pos:wave_length+start_pos]), 
                                                             np.array(wave_generated[:wave_length]))
                            if current_mse < best_mse:
                                best_mse = current_mse
                                optimal_kp2 = [x, y]
                                best_interval = interval
                                best_start_pos = start_pos
                        # except ValueError as e:
                        #     print(e)
                        #     continue
                logger.log(f"Rx[{self.id} ]Interval: {interval}, Start Position: {start_pos}, best_mse = {best_mse}")
        
        self.key_point2 = optimal_kp2
        self.key_point1[0] = init_key1_x - best_start_pos
        self.interval = best_interval
        self.start_pos = best_start_pos
        if self.key_point2 is None:
            logger.log(f"Rx[{self.id}] Unable to find a valid key point 2", level='WARNING')
            warnings.warn("Unable to find a valid key point 2")
        
        logger.log(f"Rx[{self.id}] Optimized key points: {self.key_point1}, {self.key_point2}, Best MSE: {best_mse}, Interval: {best_interval}, Start Position: {best_start_pos}")    
        logger.log(
            f"[ChannelModel_Rx#{self.id}] Training complete | "
            f"key_point1={self.key_point1}, key_point2={self.key_point2}, "
            f"mse={best_mse:.4f}, interval={best_interval}, start_pos={best_start_pos}"
        )
        return best_mse
    
    def generate_wave_prediction_old(self, bits_sequence):
        """Legacy waveform composition routine for a bit sequence."""

        if isinstance(bits_sequence,str):
            bits_sequence = BinaryUtils.binary_string_to_list(bits_sequence)
        total_length = (len(bits_sequence)-1) * self.interval + self.sig_end
        composite_signal = np.zeros(total_length)

        for i, bit in enumerate(bits_sequence):
            start = i * self.interval
            end = start + self.sig_end
            waveform = self.generate_bit_waveform(bit)
            composite_signal[start:end] += waveform
            
        return composite_signal
    
    @staticmethod
    def cal_matrix_A_b(bits_sequence, x1 , y1, x2, interval, sig_end):
        """Build the linear terms used to synthesize a waveform sequence.

        Args:
            bits_sequence (str | list[int]): Bit sequence to synthesize.
            x1 (int): First key-point position.
            y1 (float): First key-point amplitude.
            x2 (int): Second key-point position.
            interval (int): Symbol interval.
            sig_end (int): Length of the single-bit template.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]: Linear coefficient arrays
            `(A_total, b_total)` such that `A_total * y2 + b_total` yields the
            full waveform.
        """
        
        if isinstance(bits_sequence,str):
            bits_sequence = BinaryUtils.binary_string_to_list(bits_sequence)
        total_length = (len(bits_sequence)-1) * interval + sig_end
        
        A_total = np.zeros(total_length)
        b_total = np.zeros(total_length)
        
        for i, bit in enumerate(bits_sequence):
            start_index = i * interval
            if start_index >= total_length:
                break

            if bit == 0:
                continue

            A_bit = np.zeros(sig_end)
            b_bit = np.zeros(sig_end)

            if x1 > 0:
                t_values1 = np.arange(0, min(x1,sig_end))
                b_bit[t_values1] = (y1 / x1) * t_values1

            if x2 > x1 and x1 < sig_end:
                t_values2 = np.arange(x1, min(x2, sig_end))
                A_bit[t_values2] = (t_values2 - x1) / (x2 - x1)
                b_bit[t_values2] = y1 * (x2 - t_values2) / (x2 - x1)

            if x2 < sig_end:
                t_values3 = np.arange(x2, sig_end)
                A_bit[t_values3] = (sig_end - t_values3) / (sig_end - x2)

            end_index = min(sig_end, total_length - start_index)
            A_total[start_index:start_index + end_index] += A_bit[:end_index]
            b_total[start_index:start_index + end_index] += b_bit[:end_index]
        return A_total, b_total

    def generate_wave_prediction(self, bits_sequence):
        """Generate the composite waveform for a bit sequence.

        Args:
            bits_sequence (str | list[int]): Bit sequence to synthesize.

        Returns:
            numpy.ndarray: Predicted waveform for the entire sequence.
        """

        A_total, b_total = self.cal_matrix_A_b(bits_sequence, self.key_point1[0], self.key_point1[1], self.key_point2[0], self.interval, self.sig_end)
        composite_signal = A_total * self.key_point2[1] + b_total

        return composite_signal

    def generate_preamble(self):
        """Generate the predicted waveform for the configured preamble.

        Returns:
            numpy.ndarray: Predicted preamble waveform cropped to signal length.
        """
        composite_signal = self.generate_wave_prediction(self.preamble_bits)    
        return composite_signal[:len(self.yRx)]
    
    def reset_start_pos(self):
        """Trim the cached signal after a positive start-position offset."""
        if self.start_pos > 0:
            self.yRx = self.yRx[self.start_pos:]
            self.start_pos = 0

    def visualize(self):
        """Plot the received signal together with the fitted preamble waveform.

        Returns:
            matplotlib.figure.Figure: Figure showing the fit result.
        """
        signal = np.array(self.yRx)
        composite_signal = self.generate_preamble()
        start_pos = self.start_pos
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(signal, label='Received Signal')
        ax.plot(range(start_pos, start_pos + len(composite_signal)), composite_signal, label='Signal Prediction')
        ax.axvline(start_pos, color='purple', linestyle='--', label='start pos')
        ax.set_title('Stationary Points Detection and Channel Estimation')
        ax.legend()
        ax.grid(True)
        return fig


ChannelModel_Rx = ChanneModel_Rx
