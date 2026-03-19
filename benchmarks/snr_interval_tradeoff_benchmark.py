"""Benchmark BER versus SNR and symbol interval.

This script sweeps noise levels and symbol intervals, estimates BER and packet
loss, and writes per-configuration results to a CSV file.
"""

import sys
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from AdaptMolMAC import *
import AdaptMolMAC
import matplotlib.pyplot as plt
import numpy as np
import csv
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

AdaptMolMAC.Logger.LOG_ENABLED = False
MAX_WORKERS = 1

def run_BER(bit_length, interval):
    """Run one BER measurement for a given payload length and interval.

    Args:
        bit_length (int): Payload length in bits.
        interval (int): Symbol interval used by the transmitter.

    Returns:
        float: Measured bit error rate.
    """
    generator = [
        [1, 1, 1],  # 0b111 (7)
        [1, 0, 1],  # 0b101 (5)
        [1, 0, 0]   # 0b110 (6)
    ]
    Tx1 = ChannelModel_Tx(interval, 50, 1.0, n_preamble=Settings.PREAMBLE_NUM, viterbi_gen=generator)
    bitstr = ''.join(np.random.choice(['0','1'], size=bit_length))
    yRx0 = Tx1.transmit(bitstr)
    
    signal_processor = AdaptMolMAC.SignalProcessor()
    yRx = signal_processor.process_signal(yRx0, x_offset=1)
    filtered_yRx = signal_processor.kalman_filter(yRx)
    filtered_yRx = signal_processor.adaptive_threshold_filter(filtered_yRx)
    filtered_yRx = signal_processor.retain_peak_points_filter(peak_points_num= 3 + AdaptMolMAC.Settings.PEAK_POINT_EXCE_CUT, yrx_data=filtered_yRx)
    stationary_processor = AdaptMolMAC.StationaryProcessor(n_preamble=AdaptMolMAC.Settings.PREAMBLE_NUM)
    
    stationary_processor = AdaptMolMAC.StationaryProcessor(n_preamble=AdaptMolMAC.Settings.PREAMBLE_NUM)

    chnl_info1, _ = stationary_processor.estimate_channel(yRx, filtered_yRx, interval_dev=0, start_pos_dev=1)
    chnl_info1.reset_start_pos()
    decoder = AdaptMolMAC.DynamicDecoder(chnl_info1.yRx, [chnl_info1], generator)
    chnl_pred_list = decoder.decode(MAX_SIGNAL_NUM = 1)
    
    preamble_len = len(stationary_processor.preamble)
    preamble_len += len(AdaptMolMAC.Settings.CHECK_CODE)

    recv_bits = chnl_pred_list[0][preamble_len:len(yRx0.send_bits[0])]
    recv_dec_bits = AdaptMolMAC.viterbi_decode(recv_bits, generator, len(generator[0]))
    
    error_count = sum(1 for a, b in zip(bitstr, recv_dec_bits) if a != b)
    ber = error_count / bit_length
    sys.stdout = sys.__stdout__
    print(f"BER, length {bit_length}: {ber}")
    sys.stdout = open(os.devnull, 'w')
    
    return ber

def SNR_to_noise(SNR):
    """Convert an SNR value in dB to a relative noise level.

    Args:
        SNR (float): Signal-to-noise ratio in dB.

    Returns:
        float: Relative noise magnitude.
    """
    return 10 ** (-(SNR / 10))

def run_with_noise_interval(SNR, interval, times, bit_length):
    """Run repeated BER measurements for one SNR/interval pair.

    Args:
        SNR (float): Signal-to-noise ratio in dB.
        interval (int): Symbol interval to test.
        times (int): Number of repeated trials.
        bit_length (int): Payload length in bits.

    Returns:
        tuple[float, int, float, float]: Noise level, interval, packet-loss
        rate, and mean BER.
    """
    noise = SNR_to_noise(SNR)
    ChannelModel_Tx.set_noise_param(0,0,0)
    Tx1 = ChannelModel_Tx(interval, 50, 1.0)
    yRx_sample = Tx1.transmit("1")
    data_sample = yRx_sample.yRxData
    max_value = -1
    max_i = 0
    for i in range(len(data_sample)):
        if data_sample[i] > max_value:
            max_value = data_sample[i]
            max_i = i
            

    noise_value = max_value * noise
    ChannelModel_Tx.set_noise_param(0, noise_value, 0)
    
    ber_sum = 0
    success_sum = 0
    fail_sum = 0
    for i in range(times):
        try:
            ber_sum += run_BER(bit_length, interval)
            success_sum += 1
        except Exception as e:
            fail_sum += 1
            
    if success_sum == 0:
        return noise, interval, 1.0, 0.5
    return noise, interval, fail_sum/times, ber_sum/success_sum

def H(p):
    """Compute the binary entropy function.

    Args:
        p (float): Bernoulli probability.

    Returns:
        float: Binary entropy of `p`.
    """
    if p == 0 or p==1:
        return 0
    from math import log2
    res  = -p*log2(p)-(1-p)*log2(1-p)
    return res

def main():
    """Run the SNR and interval tradeoff benchmark."""
    sys.stdout = open(os.devnull, 'w')
    filename_note = "SNR_1-60,itv_1-300"
    csv_file = f"simulation_results_{filename_note}.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['noise', 'interval', 'error_rate', 'packet_loss', 'mutual_info', 'max_rate', 'timestamp'])
    
    AdaptMolMAC.simParams.set_params(T = 10)
    
    SNR_range = list(np.arange(1,60,2))
    interval_range = list(np.arange(1,300,4))
    
    MAX_WORKERS = min(MAX_WORKERS, len(interval_range))
    
    for noise in SNR_range:
        maxR = -1
        best_i = -1
        best_p = -1
        best_e = -1
        best_interval = -1
        
        with ProcessPoolExecutor(MAX_WORKERS=MAX_WORKERS) as executor:
            future_to_interval = {
                executor.submit(run_with_noise_interval, noise, interval, 50, 100): interval
                for interval in interval_range
            }
            
            for future in as_completed(future_to_interval):
                interval = future_to_interval[future]
                try:
                    noise_val, interval_val, e, p = future.result()
                    I = (1-e)*(1-H(p))
                    R = 10 * I/interval
                    
                    with open(csv_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([noise, interval_val, p, e, I, R, time.time()])
                    
                    if R > maxR:
                        best_interval = interval_val
                        maxR = R
                        best_i = I
                        best_p = p
                        best_e = e
                
                except Exception as exc:
                    print(f'Interval {interval} generated an exception: {exc}')
        
        sys.stdout = sys.__stdout__
        print(f"noise={noise}, e={best_e}%, p={best_p}%, I={best_i}bits, maxR={maxR}bits/s, best_interval={best_interval}")
        
        sys.stdout = open(os.devnull, 'w')


if __name__ == "__main__":
    main()