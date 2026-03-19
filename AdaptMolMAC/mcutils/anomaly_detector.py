"""Streaming anomaly detectors used by the decoding pipeline.

The module provides both a lightweight t-probability detector for online error
monitoring and a KL-style detector for sequence-level anomaly analysis.
"""

import numpy as np
import csv
import matplotlib.pyplot as plt
import math
from collections import deque
from scipy.stats import t
from ..config import Settings

class TProbAnomalyDetector:
    """Detect anomalies with a two-tailed t-distribution test.

    The detector maintains streaming mean and variance estimates and labels
    samples as abnormal when their probability under the current reference
    distribution falls below a configured threshold.

    Attributes:
        min_capacity (int): Minimum sample count before detection is active.
        prob_threshold (float): Two-tailed probability threshold.
        window_size (int | None): Optional sliding-window size.
        abnormal_pos (int | None): First abnormal position found during an
            initial scan.
    """
    
    def __init__(self, yRx = None, min_capacity=3, prob_threshold=Settings.DEFAULT_PROB_THRESHOLD, window_size=None):
        """Initialize the detector and optionally scan an initial sequence.

        Args:
            yRx (iterable[float] | None): Optional sequence to scan
                immediately.
            min_capacity (int): Minimum sample count before detection becomes
                active.
            prob_threshold (float): Two-tailed probability threshold.
            window_size (int | None): Optional sliding-window length.

        Raises:
            ValueError: If `min_capacity` is smaller than 3.
            ValueError: If `prob_threshold` is outside the interval `(0, 1)`.
        """
        if min_capacity < 3:
            raise ValueError("Minimum sample size must be at least 3.")
        if not 0 < prob_threshold < 1:
            raise ValueError("Probability threshold must be within (0, 1).")
        
        self.min_capacity = min_capacity
        self.prob_threshold = prob_threshold
        self.window_size = window_size
        
        self._reset_stats()
        
        self.window_queue = deque()
        self.original_count = 0
        
        self.abnormal_pos = None
        if yRx is not None:
            for index, item in enumerate(yRx):
                is_abnorml , _  = self.detect(item)
                if is_abnorml:
                    self.abnormal_pos = index
                    break
                else:
                    self.add(item)
                
    def reset(self):
        """Reset all running statistics and buffered samples."""
        self._reset_stats()
        self.window_queue = deque()
        self.original_count = 0
        self.abnormal_pos = None
    
    def _reset_stats(self):
        """Clear cached descriptive statistics."""
        self.count = 0
        self.sum_x = 0.0
        self.sum_x2 = 0.0
        self.mean = float('nan')
        self.std = float('nan')
        self.df = 0
    
    def _update_stats(self, x, remove=None):
        """Update the running moments, optionally in sliding-window mode."""
        if remove is None:
            self.count += 1
            self.sum_x += x
            self.sum_x2 += x * x
        else:
            self.sum_x = self.sum_x - remove + x
            self.sum_x2 = self.sum_x2 - remove*remove + x*x
        
        if self.count > 0:
            self.mean = self.sum_x / self.count
            
            if self.count > 1:
                variance = (self.sum_x2 - (self.sum_x**2)/self.count) / (self.count - 1)
                self.std = math.sqrt(max(variance, 0))
                self.df = self.count - 1
    
    def _calculate_probability(self, x):
        """Return the two-tailed probability of observing sample `x`."""
        if self.count < 2 or math.isnan(self.mean) or math.isnan(self.std) or self.std <= 0:
            return 1.0
        
        t_value = (x - self.mean) / (self.std)
        
        tail_prob = t.sf(abs(t_value), self.df)
        double_tail_prob = 2 * tail_prob  

        return min(max(double_tail_prob, 0.0), 1.0)

    def detect(self, x, if_add=False):
        """Classify one sample and optionally add it to the reference set.

        Args:
            x (float): Sample to evaluate.
            if_add (bool): Whether to insert the sample when it is not
                classified as abnormal.

        Returns:
            tuple[bool, float]: Anomaly flag and associated probability.
        """
        self.original_count += 1
        
        if self.count < self.min_capacity:
            if if_add:
                self._add_to_data(x)
            return False, 1.0
        
        prob = self._calculate_probability(x)
        
        if prob < self.prob_threshold:
            return True, prob
        
        if if_add:
            self._add_to_data(x)
        return False, prob
    
    def add(self, x):
        """Add one sample to the reference statistics."""
        self._add_to_data(x)
    
    def _add_to_data(self, x):
        """Insert one sample into the detector state."""
        if self.window_size is not None:
            if len(self.window_queue) == self.window_size:
                removed = self.window_queue.popleft() 
                self._update_stats(x, remove=removed)
                self.window_queue.append(x)
            else:
                self._update_stats(x)
                self.window_queue.append(x)
        else:
            self._update_stats(x)
    
    def get_statistics(self):
        """Return the current detector statistics as a dictionary."""
        return {
            'count': self.count,
            'mean': self.mean,
            'std': self.std,
            'threshold': self.prob_threshold,
            'min_capacity': self.min_capacity,
            'window_size': self.window_size,
            'total_processed': self.original_count,
            'df': self.df
        }


class KLStreamingAnomalyDetector:
    """Detect anomalies using a streaming KL-style divergence measure.

    Attributes:
        interp_factor (int): Number of interpolated points inserted between raw
            samples.
        raw_sequence (list[float]): Original appended samples.
        interp_buffer (list[float]): Interpolated working sequence.
        processed_idx (int): Index of the last processed interpolated sample.
    """
    
    def __init__(self, base_sequence=None, interpolation_factor=10):
        """Initialize detector buffers and optionally preload a baseline.

        Args:
            base_sequence (iterable[float] | None): Optional initial sequence.
            interpolation_factor (int): Number of interpolated samples inserted
                between consecutive raw observations.
        """
        self.interp_factor = interpolation_factor
        self.raw_sequence = []
        self.interp_buffer = []
        self._init_statistics()
        self.processed_idx = 0
        
        if base_sequence is not None:
            for item in base_sequence:
                self.append(item)
    
    def _init_statistics(self):
        """Reset the statistics used for the KL-style score."""
        self.sum_x = 0.0
        self.sum_x2 = 0.0
        self.count = 0
        
        self.kl_sum = 0.0
        self.kl_sum_log = 0.0
        self.kl_count = 0
    
    def _interpolate(self, new_item):
        """Insert interpolated samples between consecutive raw samples."""
        if len(self.raw_sequence) < 1:
            self.raw_sequence.append(new_item)
            self.interp_buffer.append(new_item)
            return
        
        prev_item = self.raw_sequence[-1]
        step = (new_item - prev_item) / (self.interp_factor + 1)
        
        self.interp_buffer.extend(
            prev_item + i*step 
            for i in range(1, self.interp_factor + 1)
        )
        self.interp_buffer.append(new_item)
        self.raw_sequence.append(new_item)
    
    def _update_statistics(self, x):
        """Update first and second moments for one sample."""
        self.count += 1
        self.sum_x += x
        self.sum_x2 += x**2
    
    def _compute_moments(self):
        """Compute the current mean and standard deviation."""
        mu = self.sum_x / self.count
        var = (self.sum_x2 / self.count) - mu**2
        var = max(var, 0.0)
        sigma = np.sqrt(var) if var > 0 else 0.0
        return mu, sigma
    
    def _anomaly_score(self, x, mu, sigma):
        """Convert a sample into a normalized squared-distance score."""
        if sigma == 0:
            return 0.0
        return ((x - mu) / sigma)**2
    
    def _update_kl_metrics(self, score):
        """Update the aggregated divergence terms with one score."""
        if score <= 0:
            return
        
        self.kl_count += 1
        self.kl_sum += score
        self.kl_sum_log += score * np.log(score)
    
    def kl_divergence(self):
        """Return the current divergence estimate."""
        if self.kl_sum <= 0 or self.kl_count == 0:
            return 0.0
        
        p_sum = self.kl_sum
        term1 = (self.kl_sum_log - p_sum * np.log(p_sum)) / p_sum
        term2 = np.log(self.kl_count)
        return (term1 + term2) / np.log(2)
    
    def _get_original_index(self, idx):
        """Map an interpolated-buffer index back to the raw sequence index."""
        if idx % (self.interp_factor + 1) == 0:
            return idx // (self.interp_factor + 1)
        else:
            return (idx // (self.interp_factor + 1)) + 1
    
    def detect_anomaly(self, kl_threshold=2.1, verbose=False):
        """Scan buffered samples and remove the first detected anomaly.

        Args:
            kl_threshold (float): Divergence threshold used for anomaly
                detection.
            verbose (bool): Whether to print intermediate diagnostic output.

        Returns:
            tuple[int | None, float]: Original-sequence anomaly position and the
            latest KL value.
        """
        start_idx = self.processed_idx
        end_idx = len(self.interp_buffer)
        for idx in range(start_idx, end_idx):
            x = self.interp_buffer[idx]
            self._update_statistics(x)
            mu, sigma = self._compute_moments()
            score = self._anomaly_score(x, mu, sigma)
            self._update_kl_metrics(score)
            kl_value = self.kl_divergence()
            original_pos = self._get_original_index(idx)
            if verbose:
                print(f"[KLStreamingAnomalyDetector] step={idx}, kl={kl_value:.2f}, original_index={original_pos}")
            if kl_value > kl_threshold:
                self.remove_original_point(original_pos)
                self.processed_idx = 0
                return original_pos, kl_value
        self.processed_idx = end_idx
        return None, kl_value
    
    def remove_original_point(self, original_pos):
        """Remove one raw sample and rebuild the interpolated buffers."""
        if original_pos < 0 or original_pos >= len(self.raw_sequence):
            return
        
        del self.raw_sequence[original_pos]
        
        self.interp_buffer.clear()
        self._init_statistics()
        self.processed_idx = 0
        
        temp_raw = self.raw_sequence.copy()
        self.raw_sequence = []
        for item in temp_raw:
            self.append(item)
    
    def append(self, new_item):
        """Append a new raw sample to the detector."""
        self._interpolate(new_item)
    
    def original_sequence_with_x(self):
        """Return the raw sequence together with its sample indices."""
        x_time = np.arange(len(self.raw_sequence))
        return self.raw_sequence.copy(), x_time
    
if __name__ == "__main__":
    ab_dect = TProbAnomalyDetector()
    start_pos =0
    y = []
    x = []
    with open('main/yRx.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for index,row in enumerate(reader):
            x.append(index)
            y.append(float(row[0]))
            
    for index, i in enumerate(y):
        if_chagne, value = ab_dect.append(i)
        print(f"[TProbAnomalyDetector] signal={i}, value={value}, anomaly_detected={if_chagne}")
        
        if if_chagne:
            start_pos = index
            break           
    print(f"[TProbAnomalyDetector] start_pos={start_pos}")
    # y,x = ab_dect.original_sequence_with_x()

    plt.plot(x, y)
    plt.axvline(x=start_pos, color='r', linestyle='--', label='Signal Start')
    plt.title(f'yRx Plot')
    plt.xlabel('time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()
    
