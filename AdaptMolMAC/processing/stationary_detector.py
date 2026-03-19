"""Stationary-point detection and channel-estimation utilities.

This module extracts peaks and valleys from preprocessed signals and fits the
receiver-side channel template used by downstream decoding logic.
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

from ..mcutils import logger, generate_preamble_bits
from ..config import Settings
from ..models import ChanneModel_Rx

class StationaryPoints:
    """Store detected peak and valley candidates.

    Attributes:
        points (list[tuple[int, str]]): Stored stationary points, where the
            second element marks each point as `"peak"` or `"valley"`.
    """

    def __init__(self, points=None):
        """Initialize the stationary-point container.

        Args:
            points (list[tuple[int, str]] | None): Optional initial point list.
        """
        self.points = points if points is not None else []

    def extract_peaks(self):
        """Return the positions currently labeled as peaks."""
        return [int(point[0]) for point in self.points if point[1] == 'peak']

    def extract_valleys(self):
        """Return the positions currently labeled as valleys."""
        return [int(point[0]) for point in self.points if point[1] == 'valley']

    def sort_by_position(self):
        """Sort stored stationary points by sample index."""
        self.points.sort(key=lambda x: x[0])

    def append_peak(self, point):
        """Append one peak candidate."""
        self.points.append((point, 'peak'))

    def append_valley(self, point):
        """Append one valley candidate."""
        self.points.append((point, 'valley'))


class StationaryProcessor:
    """Estimate channel timing and template parameters from stationary points.

    Attributes:
        id (int): Instance identifier used in logs.
        d2_threshold (float): Curvature threshold for peak/valley acceptance.
        n_peak (int): Number of preamble peaks required for estimation.
        n_preamble (int): Preamble order.
        preamble (str): Generated preamble bit sequence.
    """

    next_id = 0

    def __init__(self, n_preamble, d2_threshold=0, n_peak= None):
        """Configure stationary-point detection and preamble matching.

        Args:
            n_preamble (int): Preamble order used by the estimator.
            d2_threshold (float): Minimum second-derivative magnitude required
                for accepting a peak or valley.
            n_peak (int | None): Expected number of preamble peaks. If omitted,
                `n_preamble + 1` is used.
        """
        self.id = StationaryProcessor.next_id
        StationaryProcessor.next_id += 1

        self.d2_threshold = d2_threshold
        if n_peak is None:
            self.n_peak = n_preamble + 1
        else:
            self.n_peak = n_peak
        self.n_preamble = n_preamble
        self.preamble = generate_preamble_bits(n_preamble)
        logger.log(
            f"[StationaryProcessor#{self.id}] Initialized "
            f"(d2_threshold={d2_threshold}, n_peak={n_peak}, preamble={self.preamble})"
        )

    def detect(self, y_smooth):
        """Detect local peaks and valleys from a smoothed signal.

        Args:
            y_smooth (yRxData): Smoothed received signal.

        Returns:
            StationaryPoints: Detected stationary points sorted by position.
        """
        dy = np.gradient(y_smooth.yRxData)
        d2y = np.gradient(dy)

        stationary_points = StationaryPoints()
        sign_changes = np.where(np.diff(np.sign(dy)) != 0)[0]

        for i in sign_changes:
            if dy[i] > 0 and dy[i + 1] < 0:
                candidate_idx = i if y_smooth[i] > y_smooth[i + 1] else i + 1
                if d2y[candidate_idx] < -self.d2_threshold:
                    stationary_points.append_peak(candidate_idx)
            elif dy[i] < 0 and dy[i + 1] > 0:
                candidate_idx = i if y_smooth[i] < y_smooth[i + 1] else i + 1
                if d2y[candidate_idx] > self.d2_threshold:
                    stationary_points.append_valley(candidate_idx)

        stationary_points.sort_by_position()
        return stationary_points

    @staticmethod
    def compute_d_s(sum_1, sum_T, sum_T2, sum_k, sum_kT):
        """Solve the least-squares timing parameters for triangular spacing.

        Args:
            sum_1 (float): Sum of constant terms.
            sum_T (float): Sum of triangular indices.
            sum_T2 (float): Sum of squared triangular indices.
            sum_k (float): Sum of observed positions.
            sum_kT (float): Sum of position-index products.

        Returns:
            tuple[float | None, float | None]: Intercept and slope of the fitted
            timing model. Returns `(None, None)` when the system is singular.
        """
        XTX = np.array([[sum_1, sum_T], [sum_T, sum_T2]])
        det = XTX[0, 0] * XTX[1, 1] - XTX[0, 1] ** 2
        if abs(det) < 1e-6:
            return None, None
        inv_XTX = np.array([[XTX[1, 1], -XTX[0, 1]], [-XTX[0, 1], XTX[0, 0]]]) / det
        beta = inv_XTX @ np.array([sum_k, sum_kT])
        return beta[0], beta[1]

    @staticmethod
    def gen_N(d, s, n):
        """Generate an ideal triangular-index timing sequence."""
        return np.array([d + s * (i * (i + 1) // 2) for i in range(n)])

    @staticmethod
    def compute_sequence(K, n):
        """Fit one candidate peak sequence and return its residual error."""
        assert len(K) == n

        sum_1 = 1
        sum_T = 0
        sum_T2 = 0
        sum_k = K[0]
        sum_kT = 0
        sum_k_sq = K[0] ** 2

        for i in range(2, n + 1):
            T = (i - 1) * i // 2
            sum_1 += 1
            sum_T += T
            sum_T2 += T ** 2
            sum_k += K[i - 1]
            sum_kT += K[i - 1] * T
            sum_k_sq += K[i - 1] ** 2

        d, s = StationaryProcessor.compute_d_s(sum_1, sum_T, sum_T2, sum_k, sum_kT)
        if d is None or s is None:
            return None, None, np.inf

        N_tmp = StationaryProcessor.gen_N(d, s, n)
        err = np.linalg.norm(np.array(N_tmp) - np.array(K))
        return d, s, err

    @staticmethod
    def non_continuous_matching(K, n):
        """Search for the best subset matching the preamble timing pattern.

        Args:
            K (list[int]): Candidate peak positions.
            n (int): Number of peaks required by the preamble.

        Returns:
            tuple[float | None, float | None, float]: Best intercept, best
            interval slope, and fitting error.
        """
        K = sorted(K)
        if len(K) < n:
            return None, None, np.inf

        best_d, best_s, min_err = None, None, float('inf')
        for indices in combinations(range(1, len(K)), n - 1):
            indices = (0,) + indices
            sub_K = [K[i] for i in indices]
            d, s, err = StationaryProcessor.compute_sequence(sub_K, n)
            if err < min_err:
                best_d, best_s, min_err = d, s, err
            elif err == min_err and d is not None:
                if best_d is None or d < best_d or (d == best_d and s < best_s):
                    best_d, best_s = d, s
        return best_d, best_s, min_err

    def estimate_channel(self, yRx, filtered_yRx=None, interval_dev=0, start_pos_dev=0, visualize=False):
        """Estimate a receiver channel template from a received waveform.

        Args:
            yRx (yRxData | numpy.ndarray | list): Original received signal.
            filtered_yRx (yRxData | None): Optional prefiltered signal used for
                stationary-point detection.
            interval_dev (int): Search radius for interval fitting.
            start_pos_dev (int): Search radius for start-position fitting.
            visualize (bool): Whether to display the detected stationary points.

        Returns:
            tuple[ChanneModel_Rx | None, float]: Fitted channel model and its
            fitting error. If estimation fails, returns `(None, np.inf)`.
        """

        logger.log(f"[StationaryProcessor#{self.id}] Starting channel estimation")

        if filtered_yRx is None:
            filtered_yRx = yRx

        stationary_points = self.detect(filtered_yRx)
        peak_points = stationary_points.extract_peaks()

        ## DEBUG

        # pulse_signal = np.zeros_like(yRx.yRxData)
        # pulse_signal[peak_points] = 1

        # coeffs = pywt.wavedec(pulse_signal, 'db4', level=4)
        # plt.figure(figsize=(12, 8))
        # for i, c in enumerate(coeffs):
        #     plt.subplot(len(coeffs), 1, i + 1)
        #     plt.plot(c)
        #     plt.title(f'Wavelet Coefficients Level {i}')
        #     plt.grid(True)
        # plt.tight_layout()
        # plt.show()

        # exit(0)

        ##
        logger.log(f"[StationaryProcessor#{self.id}] Detected peak points: {peak_points}")

        # Use the new method to calculate interval
        if len(peak_points) < self.n_peak:
            logger.log(
                f"[StationaryProcessor#{self.id}] At least {self.n_peak} peak points are required for channel estimation",
                level="WARNING",
            )
            return None ,np.inf
            raise ValueError(f"StationaryProcessor[{self.id}]: At least {self.n_peak} peak points are required for channel estimation")
            
        d, s, err = self.non_continuous_matching(peak_points, self.n_peak)
        if d is None or s is None:
            logger.log(f"[StationaryProcessor#{self.id}] Failed to find a valid signal interval pattern", level="WARNING")
            return None ,np.inf
            raise ValueError(f"StationaryProcessor[{self.id}]: Failed to find a valid signal interval pattern")
        
        logger.log(f"[StationaryProcessor#{self.id}] Estimated parameters: d={d}, s={s}, error={err}")

        start_pos = 0
        interval = round(s)
        logger.log(f"[StationaryProcessor#{self.id}] Estimated start position={start_pos}, interval={interval}")
        
        channel_signal = ChanneModel_Rx(
            interval=interval,
            peak1_x=peak_points[0],
            yRx=yRx,
            n_preamble=self.n_preamble,
            n_peak = self.n_peak
        )

        logger.log(f"[StationaryProcessor#{self.id}] Training channel key points")
        mse = channel_signal.train_keypoint(interval_dev = interval_dev, start_pos_dev = start_pos_dev)
        if Settings.FINE_TUNE_ENABLE:
            try:
                mse = channel_signal.fine_tune_params()
            except Exception as e:
                print(f"[StationaryProcessor#{self.id}] Fine-tuning failed: {e}")
                logger.log(f"[StationaryProcessor#{self.id}] Fine-tuning failed: {e}", level="WARNING")

        logger.log(f"[StationaryProcessor#{self.id}] Channel key-point training finished")
        start_pos = channel_signal.start_pos
        interval = channel_signal.interval

        logger.log(f"[StationaryProcessor#{self.id}] Optimized parameters: start_pos={start_pos}, interval={interval}")
        logger.log(yRx, point_list=stationary_points.extract_peaks(),
                   description=f"[StationaryProcessor#{self.id}] Received signal with stationary points",
                   img_save_subfolder="stationary_points")
        logger.log("")

        if visualize:
            self._plot_estimation(yRx.yRxData, stationary_points)

        return channel_signal, mse

    def _plot_estimation(self, signal, points):
        """Visualize detected peaks and valleys on the input signal.

        Args:
            signal (numpy.ndarray): Signal to plot.
            points (StationaryPoints): Detected stationary points.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(signal, label='Received Signal')
        # plt.plot(range(start_pos, start_pos + len(signal)),signal, label='Received Signal')

        # plt.plot(composite_signal, label = 'Signal Prediction')

        peak_x = points.extract_peaks()
        valley_x = points.extract_valleys()
        plt.scatter(peak_x, signal[peak_x], color='red', label='peak', zorder=5)
        plt.scatter(valley_x, signal[valley_x], color='blue', label='valley', zorder=5)

        plt.title('Stationary Points Detection and Channel Estimation')
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    t = np.linspace(0, 10, 500)
    yRx = np.sin(t * 3) + np.random.normal(0, 0.2, 500)

    processor = StationaryProcessor(d2_threshold=0.1)
    try:
        model = processor.estimate_channel(yRx, start_pos=50)
        plt.plot(model.generate_bit_waveform(1), label='bit1 wave')
        plt.plot(model.generate_bit_waveform(0), label='bit0 wave')
        plt.title('Optimized signal waveform')
        plt.legend()
        # plt.show()
    except ValueError as e:
        print(f"Processing failed: {str(e)}")
