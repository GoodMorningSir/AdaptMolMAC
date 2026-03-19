"""Signal preprocessing routines used before channel estimation and decoding.

The functions in this module smooth, threshold, and crop received signals so
that downstream estimators operate on a cleaner representation.
"""

from ..mcutils import TProbAnomalyDetector, logger
from ..models import yRxData

from pykalman import KalmanFilter
from skimage.filters import threshold_otsu
import numpy as np

class SignalProcessor:
    """Preprocess received signals for estimation and dynamic decoding.

    The processor groups together the main filtering utilities used by the
    package, including baseline correction, Kalman smoothing, thresholding, and
    peak-based cropping.

    Attributes:
        id (int): Instance identifier used in logs.
        kl_threshold (float): Reserved anomaly threshold configuration.
    """

    next_id  = 0

    def __init__(self, kl_threshold: float = 2.0):
        """Initialize a signal processor.

        Args:
            kl_threshold (float): Threshold configuration stored on the
                processor instance.
        """
        self.id = SignalProcessor.next_id
        SignalProcessor.next_id += 1
        
        self.kl_threshold = kl_threshold
        logger.log(f"[SignalProcessor#{self.id}] Initialized with kl_threshold={kl_threshold}")
        

    def process_signal(self, yrx_data, x_offset = 1) -> yRxData:
        """Baseline-correct a received signal using anomaly-based alignment.

        Args:
            yrx_data (yRxData): Input received signal container.
            x_offset (int): Number of samples to step back from the detected
                anomaly position.

        Returns:
            yRxData: Baseline-corrected signal data.

        Raises:
            ValueError: If the input signal is not one-dimensional.
        """
        logger.log(f"[SignalProcessor#{self.id}] Processing signal with x_offset={x_offset}")
        raw_signal = yrx_data.yRxData
        
        if raw_signal.ndim != 1:
            raise ValueError(f"SignalProcessor[{self.id}]: Input signal must be a 1-dimensional array")

        detector = TProbAnomalyDetector(raw_signal)
        change_point = detector.abnormal_pos

        if change_point is None:
            corrected = raw_signal - np.mean(raw_signal)
            logger.log(f"[SignalProcessor#{self.id}] No anomaly detected; returning baseline-corrected signal")
            return yRxData(raw_signal, yrx_data.send_bits, ifLogger=False)

        else:
            start_idx = max(change_point - x_offset, 0)
            baseline = np.mean(raw_signal[:start_idx])
            corrected = raw_signal - baseline
            logger.log(
                f"[SignalProcessor#{self.id}] Anomaly detected at index {change_point}; "
                f"baseline={baseline}, start_idx={start_idx}"
            )
            logger.log(
                raw_signal[:int(start_idx*2)],
                point_list=[start_idx],
                description=f"[SignalProcessor#{self.id}] Raw signal around detected start",
                img_save_subfolder="signal_processor",
            )
            logger.log(
                corrected[start_idx:],
                description=f"[SignalProcessor#{self.id}] Corrected signal for bits={yrx_data.send_bits}",
                img_save_subfolder="signal_processor",
            )
            return yRxData(process_data=corrected[start_idx:], send_bits=yrx_data.send_bits, ifLogger=False)

    def kalman_filter(
        self,
        yrx_data,
        initial_state_mean: float = 0.0,
        observation_dimension: int = 1,
        transition_covariance: float = 0.01,
        observation_covariance: float = 0.1, 
        initial_state_covariance: float = 1.0,
    ) -> yRxData:
        """Smooth a one-dimensional signal with a Kalman filter.

        Args:
            yrx_data (yRxData | list | numpy.ndarray): Signal to smooth.
            initial_state_mean (float): Initial latent-state mean.
            observation_dimension (int): Observation dimension passed to
                `pykalman`.
            transition_covariance (float): State transition covariance.
            observation_covariance (float): Observation covariance.
            initial_state_covariance (float): Initial state covariance.

        Returns:
            yRxData: Smoothed signal container.

        Raises:
            TypeError: If `yrx_data` is not a supported signal container.
        """
        logger.log(
            f"[SignalProcessor#{self.id}] Applying Kalman filter "
            f"(initial_state_mean={initial_state_mean}, observation_dimension={observation_dimension})"
        )
        if isinstance(yrx_data, (list, np.ndarray)):
            yrx_data = yRxData(process_data=np.array(yrx_data), send_bits="Unknown", ifLogger=False)
        else:
            if not isinstance(yrx_data, yRxData):
                raise TypeError(f"SignalProcessor[{self.id}]: yrx_data must be a list, numpy array, or yRxData instance")
        raw_signal = yrx_data.yRxData
        
        kalman_filter = KalmanFilter(
            initial_state_mean=initial_state_mean,
            n_dim_obs=observation_dimension,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
            initial_state_covariance=initial_state_covariance,
            transition_matrices=[1], 
            observation_matrices=[1]
        )
        
        smoothed_states, _ = kalman_filter.smooth(raw_signal)
        
        logger.log(f"[SignalProcessor#{self.id}] Kalman filter applied")
        logger.log(
            smoothed_states,
            description=f"[SignalProcessor#{self.id}] Smoothed signal after Kalman filtering",
            img_save_subfolder="kalman_filter",
        )
        return yRxData(process_data=smoothed_states.squeeze(), send_bits=yrx_data.send_bits, ifLogger=False)
    
    def percentile_threshold_filter(
        self,
        yrx_data,
        peek_pct: float = 95.0,
        threshold_pct: float = 0.5
    ) -> yRxData:
        """Suppress samples below a percentile-based threshold.

        Args:
            yrx_data (yRxData | list | numpy.ndarray): Signal to threshold.
            peek_pct (float): Percentile used to define the reference level.
            threshold_pct (float): Fraction of the percentile value used as the
                final threshold.

        Returns:
            yRxData: Thresholded signal container.

        Raises:
            TypeError: If `yrx_data` is not a supported signal container.
            ValueError: If the input signal is not one-dimensional.
        """
        logger.log(
            f"[SignalProcessor#{self.id}] Applying percentile threshold filter "
            f"(peek_pct={peek_pct}, threshold_pct={threshold_pct})"
        )
        if isinstance(yrx_data, (list, np.ndarray)):
            yrx_data = yRxData(process_data=np.array(yrx_data), send_bits="Unknown", ifLogger=False)
        else:
            if not isinstance(yrx_data, yRxData):
                raise TypeError(f"SignalProcessor[{self.id}]: yrx_data must be a list, numpy array, or yRxData instance")
        raw_signal = yrx_data.yRxData
        
        if raw_signal.ndim != 1:
            raise ValueError(f"SignalProcessor[{self.id}]: Input signal must be a 1-dimensional array")
        
        base_threshold = np.percentile(raw_signal, peek_pct)
        actual_threshold = threshold_pct * base_threshold
        filtered_signal = np.where(raw_signal >= actual_threshold, raw_signal, 0)
        
        logger.log(
            f"[SignalProcessor#{self.id}] Percentile threshold filter applied "
            f"(base_threshold={base_threshold}, actual_threshold={actual_threshold})"
        )
        logger.log(
            filtered_signal,
            description=f"[SignalProcessor#{self.id}] Signal after percentile threshold filtering",
            img_save_subfolder="percentile_threshold",
        )
        return yRxData(process_data=filtered_signal, send_bits=yrx_data.send_bits, ifLogger=False)
    
    def adaptive_threshold_filter(
        self,
        yrx_data
    ) -> yRxData:
        """Apply Otsu thresholding to keep the strongest samples.

        Args:
            yrx_data (yRxData | list | numpy.ndarray): Signal to threshold.

        Returns:
            yRxData: Thresholded signal container.

        Raises:
            TypeError: If `yrx_data` is not a supported signal container.
            ValueError: If the input signal is not one-dimensional.
        """
        logger.log(f"[SignalProcessor#{self.id}] Applying adaptive threshold filter using Otsu's method")
        if isinstance(yrx_data, (list, np.ndarray)):
            yrx_data = yRxData(process_data=np.array(yrx_data), send_bits="Unknown", ifLogger=False)
        else:
            if not isinstance(yrx_data, yRxData):
                raise TypeError(f"SignalProcessor[{self.id}]: yrx_data must be a list, numpy array, or yRxData instance")
        raw_signal = yrx_data.yRxData
        
        if raw_signal.ndim != 1:
            raise ValueError(f"SignalProcessor[{self.id}]: Input signal must be a 1-dimensional array")
        
        threshold = threshold_otsu(raw_signal)
        filtered_signal = np.where(raw_signal >= threshold, raw_signal, 0)
        
        logger.log(f"[SignalProcessor#{self.id}] Adaptive threshold filter applied with threshold={threshold}")
        logger.log(
            filtered_signal,
            description=f"[SignalProcessor#{self.id}] Signal after adaptive threshold filtering",
            img_save_subfolder="adaptive_threshold",
        )
        return yRxData(process_data=filtered_signal, send_bits=yrx_data.send_bits, ifLogger=False)
    
    def retain_peak_points_filter(
        self,
        peak_points_num,
        yrx_data
    ):
        """Crop a signal after a fixed number of detected peaks.

        Args:
            peak_points_num (int): Number of peaks to retain before cropping.
            yrx_data (yRxData | list | numpy.ndarray): Signal to crop.

        Returns:
            yRxData: Cropped signal container. If too few peaks are found, the
            original signal is returned.

        Raises:
            TypeError: If `yrx_data` is not a supported signal container.
            ValueError: If the input signal is not one-dimensional.
        """
        logger.log(f"[SignalProcessor#{self.id}] Retaining peak points (peak_points_num={peak_points_num})")
        if isinstance(yrx_data, (list, np.ndarray)):
            yrx_data = yRxData(process_data=np.array(yrx_data), send_bits="Unknown", ifLogger=False)
        else:
            if not isinstance(yrx_data, yRxData):
                raise TypeError(f"SignalProcessor[{self.id}]: yrx_data must be a list, numpy array, or yRxData instance")
        raw_signal = yrx_data.yRxData

        if raw_signal.ndim != 1:
            raise ValueError(f"SignalProcessor[{self.id}]: Input signal must be a 1-dimensional array")

        from scipy.signal import find_peaks
        peaks, _ = find_peaks(raw_signal)
        
        if len(peaks) < peak_points_num:
            logger.log(
                f"[SignalProcessor#{self.id}] Only found {len(peaks)} peaks while "
                f"{peak_points_num} were requested; returning the original signal"
            )
            return yRxData(process_data=raw_signal, send_bits=yrx_data.send_bits, ifLogger=False)
        
        nth_peak_pos = peaks[peak_points_num] 
        
        cropped_signal = raw_signal[:nth_peak_pos]
        
        logger.log(
            f"[SignalProcessor#{self.id}] Cropped signal to index range [0, {nth_peak_pos}) "
            f"using the {peak_points_num}th peak"
        )
        logger.log(
            cropped_signal,
            description=f"[SignalProcessor#{self.id}] Cropped signal with {peak_points_num} peaks",
            img_save_subfolder="peak_points_filter",
        )
        
        return yRxData(process_data=cropped_signal, send_bits=yrx_data.send_bits, ifLogger=False)
