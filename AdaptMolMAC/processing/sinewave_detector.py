"""Sine-wave-based correlation experiments for preamble detection.

This module contains an alternative experimental detector that matches a
synthetic sine-wave preamble against a received signal by correlation.
"""

import numpy as np
from ..mcutils import logger,BinaryUtils,Plotter
import matplotlib.pyplot as plt

class SineWaveGenerate:
    """Generate a sine-wave template from a binary sequence.

    Attributes:
        bits_seq (str): Bit sequence used to construct the waveform.
        k_corr (float): Linear correction factor applied across the waveform.
        amp (float): Amplitude of the sine segments.
    """

    def __init__(self, bits_seq, amp, k_corr = 0):
        """Initialize the sine-wave generator.

        Args:
            bits_seq (str): Bit sequence used to build the waveform.
            amp (float): Amplitude of the sine segments.
            k_corr (float): Optional linear correction term.

        Raises:
            AssertionError: If `bits_seq` is not a string.
        """
        BinaryUtils.validate_binary_string(bits_seq)
        assert(type(bits_seq) == str), "Input bits_seq must be a 0-1 string"
        self.bits_seq = bits_seq
        self.k_corr = k_corr
        self.amp = amp
            
    def generate_wave(self, interval):
        """Generate the waveform for the configured bit sequence.

        Args:
            interval (int): Number of samples used for each bit segment.

        Returns:
            numpy.ndarray: Generated sine-wave-based waveform.

        Raises:
            ValueError: If the bit sequence contains unsupported symbols.
        """
        t = np.linspace(0, np.pi, interval)
        one_segment = self.amp * np.sin(t)
        zero_segment = np.zeros(interval)

        segments = [one_segment if bit == '1' else zero_segment if bit == '0' else None for bit in self.bits_seq]
        if any(seg is None for seg in segments):
            raise ValueError("Invalid bit value. Only '0' and '1' are allowed.")

        wave = np.concatenate(segments)

        wave += (self.k_corr / interval) * np.arange(wave.shape[0])

        return wave

class SineWaveCorrelationProcessor:
    """Estimate channel timing by correlating against a sine-wave preamble.

    Attributes:
        id (int): Instance identifier used in logs.
        n_preamble (int): Preamble order used to generate the template.
        preamble (str): Generated preamble bit sequence.
        wave_gen (SineWaveGenerate): Template generator.
    """

    next_id = 0
    def __init__(self, n_preamble = 3, amp = 1, k_corr = 0):
        """Initialize the sine-wave correlation processor.

        Args:
            n_preamble (int): Preamble order used to build the template.
            amp (float): Template amplitude.
            k_corr (float): Optional template correction factor.
        """
        self.id = SineWaveCorrelationProcessor.next_id
        SineWaveCorrelationProcessor.next_id += 1
        
        self.n_preamble = n_preamble
        self.preamble = BinaryUtils.generate_preamble_bits(n_preamble)
        self.wave_gen = SineWaveGenerate(self.preamble, amp=amp, k_corr=k_corr)
    
    def estimate_channel(self, yRx, filtered_yRx = None, interval_range = [5,50], start_pos_dev = 0, visualize=False):
        """Estimate channel timing by brute-force correlation search.

        Args:
            yRx (yRxData): Original received signal.
            filtered_yRx (yRxData | None): Optional filtered signal used for
                correlation.
            interval_range (list[int]): Inclusive search range for the symbol
                interval.
            start_pos_dev (int): Maximum start position offset to test.
            visualize (bool): Whether to show the matched waveform.

        Returns:
            tuple[int, float, int]: Best interval, best correlation score, and
            best start position.
        """
        logger.log(f"[SineWaveCorrelationProcessor#{self.id}] Starting channel estimation")
        if filtered_yRx is None:
            filtered_yRx = yRx
        
        best_corr = -np.inf
        best_interval = None
        best_pos = None

        for interval in range(interval_range[0], interval_range[1] + 1):
            wave = self.wave_gen.generate_wave(interval)
            if len(wave) > len(filtered_yRx.yRxData):
                continue
            for pos in range(start_pos_dev + 1):
                if pos + len(wave) > len(filtered_yRx.yRxData):
                    break
                segment = filtered_yRx.yRxData[pos:pos + len(wave)]
                corr = np.dot(wave, segment) / len(wave)
                print(f"[SineWaveCorrelationProcessor#{self.id}] interval={interval}, position={pos}, correlation={corr:.4f}")
            if corr > best_corr:
                best_corr = corr
                best_interval = interval
                best_pos = pos
                
        print(f"[SineWaveCorrelationProcessor#{self.id}] best_interval={best_interval}, best_corr={best_corr}, best_pos={best_pos}")
        best_wave = self.wave_gen.generate_wave(best_interval)
        fig = Plotter.draw_originPic_predictPic(origin_signal=filtered_yRx.yRxData, predict_signal=best_wave, Point_list=[best_pos, best_pos+len(best_wave)])
        plt.show()
        exit(0)
        return best_interval, best_corr, best_pos
        
        
if __name__ == "__main__":
    
    processor = SineWaveCorrelationProcessor()
    print(f"[SineWaveCorrelationProcessor] processor_id={processor.id}")
    print(f"[SineWaveCorrelationProcessor] preamble={processor.preamble}")
    print(f"[SineWaveCorrelationProcessor] wave={processor.wave}")
    plt.plot(processor.wave)
    plt.title("Generated Sine Wave Preamble")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.show()
