"""Command-line entry point for AdaptMolMAC experiments.

This script builds one or more transmitters from CLI arguments, runs the full
processing and decoding pipeline, and prints markers consumed by the benchmark
scripts under `tests/`.
"""

import random
import numpy as np
import argparse
import json
import time

import AdaptMolMAC
from AdaptMolMAC import logger

DEBUG = False
MAX_TRANSMITTERS = 10
logger.disable()


def get_run_params():
    """Parse command-line arguments and build a normalized run configuration.

    Returns:
        dict: Parsed generator, channel parameters, payloads, and seed value.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--generator', type=str, default=None)
    for i in range(1, MAX_TRANSMITTERS + 1):
        parser.add_argument(f'--chan_para{i}', type=str, default=None)
    for i in range(1, MAX_TRANSMITTERS + 1):
        parser.add_argument(f'--tran_data{i}', type=str, default=None)
    args, unknown = parser.parse_known_args()
    def parse_json_or_default(val, default):
        if val is not None:
            return json.loads(val)
        return default
    params = {
        'seed': args.seed if args.seed is not None else 0,
        'generator': parse_json_or_default(args.generator, []),
    }
    for i in range(1, MAX_TRANSMITTERS + 1):
        param_name = f'chan_para{i}'
        params[param_name] = parse_json_or_default(getattr(args, param_name), [])
    for i in range(1, MAX_TRANSMITTERS + 1):
        param_name = f'tran_data{i}'
        params[param_name] = getattr(args, param_name) if getattr(args, param_name) is not None else ''
    
    return params

def random_bitstr(length):
    """Generate a random binary string.

    Args:
        length (int): Requested bit-string length.

    Returns:
        str: Randomly generated binary payload.
    """
    return ''.join(random.choice('01') for _ in range(length))


def print_stage(stage_name):
    """Print a standardized stage header.

    Args:
        stage_name (str): Human-readable stage name.
    """
    print(f"[Main] {stage_name}")


def log_stage(stage_name):
    """Log a standardized stage header.

    Args:
        stage_name (str): Human-readable stage name.
    """
    logger.log(f"[Main] {stage_name}")

def run():
    """Execute the full simulation, estimation, and decoding pipeline.

    Returns:
        tuple[list[str], list[str], list[float], int]: Transmitted payloads,
        decoded channel strings, accuracy values, and transmitter count.
    """
    # In debug mode, use built-in sample data for quick local validation.
    if DEBUG:
        seed = random.randint(0, 2 ** 32 - 1)
        seed = 777628
        random.seed(seed)
        np.random.seed(seed)
        logger.log(f"[Main] Random seed set to {seed}")
        generator = [
            [1, 1, 1],  # 0b111 (7)
            [1, 0, 1],  # 0b101 (5)
            [1, 0, 0]   # 0b110 (6)
        ]
        tran_data = {
            1: random_bitstr(11),
            2: random_bitstr(11),
            # 3: random_bitstr(11)
        }
        chan_para = {
            1: [17, 50, 1.0],    # interval, tx_offset, amplitude
            2: [22, 864, 1.0],   # interval, tx_offset, amplitude
            # 3: [17, 800, 1.0]
        }
    else:
        print_stage("Loading runtime parameters")
        # In non-debug mode, load experiment settings from CLI arguments.
        params = get_run_params()
        seed = params['seed']
        generator = params['generator']
        chan_para = {}
        tran_data = {}
        for i in range(1, MAX_TRANSMITTERS + 1):

            chan_key = f'chan_para{i}'
            data_key = f'tran_data{i}'
            if params.get(chan_key) and params.get(data_key):
                chan_para[i] = params[chan_key]
                tran_data[i] = params[data_key]

    start_time = time.time()
    logger.log(f"[Main] Start time: {start_time}")

    # Build all transmitters from the configured channel parameters.
    print_stage("Initializing transmitters")
    log_stage("Initializing transmitters")

    transmitters = {}
    for idx, params in chan_para.items():
        if len(params) >= 3:  # Ensure the parameter triplet is complete.
            transmitters[idx] = AdaptMolMAC.ChannelModel_Tx(
                params[0], 
                params[1], 
                params[2], 
                n_preamble=AdaptMolMAC.Settings.PREAMBLE_NUM, 
                viterbi_gen=generator
            )

    # Transmit all bit streams and superimpose their received waveforms.
    print_stage("Transmitting signals")
    log_stage("Transmitting signals")
    all_signals = []
    yRx0 = AdaptMolMAC.yRxData()
    for idx, tx in transmitters.items():
        if idx in tran_data:
            signal = tx.transmit(tran_data[idx])
            all_signals.append(signal)
            yRx0 += signal

    # Apply baseline correction, smoothing, thresholding, and peak retention.
    print_stage("Processing received signal")
    log_stage("Processing received signal")
    signal_processor = AdaptMolMAC.SignalProcessor()
    yRx = signal_processor.process_signal(yRx0, x_offset=1)
    # yRx.visualize()
    filtered_yRx = signal_processor.kalman_filter(yRx)
    filtered_yRx = signal_processor.adaptive_threshold_filter(filtered_yRx)
    filtered_yRx = signal_processor.retain_peak_points_filter(peak_points_num= 3 + AdaptMolMAC.Settings.PEAK_POINT_EXCE_CUT, yrx_data=filtered_yRx)

    stationary_processor = AdaptMolMAC.StationaryProcessor(n_preamble=AdaptMolMAC.Settings.PREAMBLE_NUM)

    # Estimate the channel model from the detected preamble waveform.
    chnl_info1, _ = stationary_processor.estimate_channel(yRx, filtered_yRx, interval_dev=0, start_pos_dev=1)
    chnl_info1.reset_start_pos()
    # chnl_info1.visualize()

    # Decode the overlapping received signals into per-channel bit sequences.
    print_stage("Decoding channels")
    log_stage("Decoding channels")
    decoder = AdaptMolMAC.DynamicDecoder(chnl_info1.yRx, [chnl_info1], generator)
    chnl_pred_list = decoder.decode(MAX_SIGNAL_NUM =len(all_signals))


    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.log(f"[Main] End time: {end_time}")
    logger.log(f"[Main] Elapsed time: {elapsed_time:.4f} seconds")
    print(f"[Main] Elapsed time: {elapsed_time:.4f} seconds")


    # Report accuracy using the encoded bit streams first.
    print_stage("Evaluating encoded-bit accuracy")
    log_stage("Evaluating encoded-bit accuracy")

    if chnl_pred_list is None:
        print("[Main] Channel prediction list is empty.")
        exit(0)
    for i, bits in enumerate(chnl_pred_list):
        print(f"[Main] Channel {i}: {bits}")

    # Prepare encoded transmit and receive bit streams for comparison.
    send_bits_list = []
    recv_bits_list = []

    for i, (idx, tx) in enumerate(transmitters.items()):
        if idx in tran_data:
            send_bits_list.append(tx.encode_data[0])  # Encoded bit stream sent by the transmitter.
            # Pull the corresponding decoded stream in the same channel order.
            if i < len(chnl_pred_list):
                recv_bits_list.append(chnl_pred_list[i])

    # Collect accuracy metrics for encoded and payload-level comparisons.
    acc_rates = []
    # Compare the encoded bit streams directly.
    acc_rates.append(AdaptMolMAC.ErrorDetect(seed, *[item for pair in zip(send_bits_list, recv_bits_list) for item in pair]).compare_accuracy())

    # Remove the preamble/check code, then run Viterbi decoding on the payload bits.
    print_stage("Evaluating payload accuracy")
    log_stage("Evaluating payload accuracy")

    dec_send_bits_list = []
    dec_recv_bits_list = []

    for i, (idx, tx) in enumerate(transmitters.items()):
        if idx in tran_data:
            dec_send_bits_list.append(tran_data[idx])
            if i < len(chnl_pred_list):
                preamble_len = len(stationary_processor.preamble)
                preamble_len += len(AdaptMolMAC.Settings.CHECK_CODE)
                recv_dec_bits = chnl_pred_list[i][preamble_len:len(tx.encode_data[0])]
                dec_recv_bits_list.append(AdaptMolMAC.viterbi_decode(recv_dec_bits, generator, len(generator[0])))
    acc_rates.append(AdaptMolMAC.ErrorDetect(seed, *[item for pair in zip(dec_send_bits_list, dec_recv_bits_list) for item in pair]).compare_accuracy())

    acc_rates = [x for sub in acc_rates for x in sub]
    # Emit machine-readable metrics for batch scripts.
    print(f"ACC_RATES_START:{json.dumps(acc_rates)}:ACC_RATES_END")
    # Emit decoded channel counts so batch scripts can detect success/failure.
    tx_count = len(transmitters)
    chnl_pred_count = len(chnl_pred_list) if chnl_pred_list is not None else 0
    print(f"TX_COUNT_START:{tx_count}:TX_COUNT_END")
    print(f"CHNL_PRED_COUNT_START:{chnl_pred_count}:CHNL_PRED_COUNT_END")

if __name__ == "__main__":
    run()