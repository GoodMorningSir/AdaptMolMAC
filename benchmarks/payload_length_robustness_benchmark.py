"""Evaluate decoding robustness as payload length increases.

This benchmark repeatedly launches the project entry point for a two-transmitter
scenario while sweeping payload bit length. It measures whether both channels
are recovered successfully and records encoded/decoded accuracy statistics.
"""

import subprocess
import random
import numpy as np
import json
import os
import sys
import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
CLI_MODULE = "AdaptMolMAC.cli"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import AdaptMolMAC
import traceback

times_per_length = 500
bit_lengths = [50, 62, 75, 100, 126, 200, 254, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500, 2000]

length_stats = {}

def random_bitstr(length):
    """Generate a random binary string.

    Args:
        length (int): Requested bit-string length.

    Returns:
        str: Randomly generated binary payload.
    """
    return ''.join(random.choice('01') for _ in range(length))

def parse_cli_output(output_text):
    """Parse CLI output for accuracy and channel-count markers.

    Args:
        output_text (str): Captured standard output from `AdaptMolMAC.cli`.

    Returns:
        tuple[list[float] | None, int | None, int | None]: Parsed accuracy
        values, transmitter count, and decoded-channel count.
    """
    try:
        lines = output_text.strip().split('\n')
        acc_rates = None
        dec_acc_rates = None
        tx_count = None
        chnl_pred_count = None
        
        for line in lines:
            if 'ACC_RATES_START:' in line and ':ACC_RATES_END' in line:
                start_marker = 'ACC_RATES_START:'
                end_marker = ':ACC_RATES_END'
                start_idx = line.find(start_marker) + len(start_marker)
                end_idx = line.find(end_marker)
                if start_idx != -1 and end_idx != -1:
                    acc_rates_json = line[start_idx:end_idx]
                    acc_rates = json.loads(acc_rates_json)
            
            if 'TX_COUNT_START:' in line and ':TX_COUNT_END' in line:
                start_marker = 'TX_COUNT_START:'
                end_marker = ':TX_COUNT_END'
                start_idx = line.find(start_marker) + len(start_marker)
                end_idx = line.find(end_marker)
                if start_idx != -1 and end_idx != -1:
                    tx_count = int(line[start_idx:end_idx])
            
            if 'CHNL_PRED_COUNT_START:' in line and ':CHNL_PRED_COUNT_END' in line:
                start_marker = 'CHNL_PRED_COUNT_START:'
                end_marker = ':CHNL_PRED_COUNT_END'
                start_idx = line.find(start_marker) + len(start_marker)
                end_idx = line.find(end_marker)
                if start_idx != -1 and end_idx != -1:
                    chnl_pred_count = int(line[start_idx:end_idx])
        
        return acc_rates, tx_count, chnl_pred_count
    except Exception as e:
        AdaptMolMAC.logger.log(f"Error parsing output: {e}")
        return None, None, None

def write_results_to_csv():
    """Create the summary CSV file and write its header row."""
    csv_path = os.path.join(os.path.dirname(__file__), 'bit_length_analysis.csv')
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        headers = [
            'Bit_Length', 'Total_Tests', 'Success_Times', 'Fail_Times', 'Success_Rate(%)',
            'Avg_tx1_send_accuracy(%)', 'Avg_tx2_send_accuracy(%)', 
            'Avg_tx1_recv_accuracy(%)', 'Avg_tx2_recv_accuracy(%)'
        ]
        writer.writerow(headers)
    
    print(f"CSV file created with headers: {csv_path}")

def write_result_row(bit_length, stats):
    """Append one bit-length summary row to the CSV file.

    Args:
        bit_length (int): Payload length represented by the row.
        stats (dict): Aggregated statistics for the current payload length.
    """
    csv_path = os.path.join(os.path.dirname(__file__), 'bit_length_analysis.csv')
    
    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        total_tests = stats['success_times'] + stats['fail_times']
        success_rate = (stats['success_times'] / total_tests * 100) if total_tests > 0 else 0
        
        row = [bit_length, total_tests, stats['success_times'], stats['fail_times'], f"{success_rate:.2f}"]
        
        if stats['accuracy_list']:
            accuracy_array = np.array(stats['accuracy_list'])
            avg_accuracy = accuracy_array.mean(axis=0)
            row.extend([
                f"{avg_accuracy[0]:.2f}", f"{avg_accuracy[1]:.2f}",
                f"{avg_accuracy[2]:.2f}", f"{avg_accuracy[3]:.2f}"
            ])
        else:
            row.extend(["0.00", "0.00", "0.00", "0.00"])
        
        writer.writerow(row)
        print(f"Result for bit length {bit_length} written to CSV")

def main():
    """Run the payload-length robustness benchmark."""
    total_iterations = len(bit_lengths) * times_per_length
    current_iteration = 0
    
    for bit_length in bit_lengths:
        length_stats[bit_length] = {
            'success_times': 0,
            'fail_times': 0,
            'accuracy_list': []
        }
    
    write_results_to_csv()
    
    for bit_length in bit_lengths:
        print(f"\n=== Testing Bit Length: {bit_length} ===")
        
        for i in range(times_per_length):
            current_iteration += 1
            
            seed = random.randint(0, 2 ** 32 - 1)
            random.seed(seed)
            np.random.seed(seed)

            print("### Test Data ###")
            generator = [
                [1, 1, 1],  # 0b111 (7)
                [1, 0, 1],  # 0b101 (5)
                [1, 0, 0]  # 0b110 (6)
            ]
            tran_data1 = random_bitstr(bit_length)
            tran_data2 = random_bitstr(bit_length)
            chan_para1 = [11, 50, 1.0]    # interval, tx_offset, amplitude
            chan_para2 = [15, 253, 1.0]   # interval, tx_offset, amplitude

            print(f'Running {CLI_MODULE}, iteration {current_iteration}/{total_iterations}, bit_length: {bit_length}, seed {seed}')
            args = [
                sys.executable, "-m", CLI_MODULE,
                '--seed', str(seed),
                '--generator', json.dumps(generator),
                '--chan_para1', json.dumps(chan_para1),
                '--chan_para2', json.dumps(chan_para2),
                '--tran_data1', tran_data1,
                '--tran_data2', tran_data2
            ]
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                encoding='utf-8',
                cwd=str(PROJECT_ROOT),
            )
            acc_rates, tx_count, chnl_pred_count = parse_cli_output(result.stdout)
            if tx_count is not None and chnl_pred_count is not None and tx_count == chnl_pred_count:
                length_stats[bit_length]['success_times'] += 1
                if acc_rates:
                    length_stats[bit_length]['accuracy_list'].append(acc_rates)
            else:
                length_stats[bit_length]['fail_times'] += 1
        
        stats = length_stats[bit_length]
        total_tests = stats['success_times'] + stats['fail_times']
        success_rate = (stats['success_times'] / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\n--- Bit Length {bit_length} Results ---")
        print(f"Success times: {stats['success_times']}")
        print(f"Fail times: {stats['fail_times']}")
        print(f"Success rate: {success_rate:.2f}%")
        
        if stats['accuracy_list']:
            accuracy_array = np.array(stats['accuracy_list'])
            avg_accuracy = accuracy_array.mean(axis=0)
            print(f"tx1_send_accuracy: {avg_accuracy[0]:.2f}%")
            print(f"tx2_send_accuracy: {avg_accuracy[1]:.2f}%")
            print(f"tx1_recv_accuracy: {avg_accuracy[2]:.2f}%")
            print(f"tx2_recv_accuracy: {avg_accuracy[3]:.2f}%")
        
        write_result_row(bit_length, stats)
    
    total_success = sum(stats['success_times'] for stats in length_stats.values())
    total_fail = sum(stats['fail_times'] for stats in length_stats.values())
    total_tests = total_success + total_fail
    
    print(f"\n=== Overall Statistics ===")
    print(f"Total Success times: {total_success}")
    print(f"Total Fail times: {total_fail}")
    print(f"Overall Success Rate: {(total_success / total_tests * 100):.2f}%")
    
    all_accuracy_lists = []
    for stats in length_stats.values():
        if stats['accuracy_list']:
            all_accuracy_lists.extend(stats['accuracy_list'])
    
    overall_avg_accuracy = np.array(all_accuracy_lists).mean(axis=0)
    print(f"\n=== Overall Average Accuracy ===")
    print(f"Overall tx1_send_accuracy: {overall_avg_accuracy[0]:.2f}%")
    print(f"Overall tx2_send_accuracy: {overall_avg_accuracy[1]:.2f}%")
    print(f"Overall tx1_recv_accuracy: {overall_avg_accuracy[2]:.2f}%")
    print(f"Overall tx2_recv_accuracy: {overall_avg_accuracy[3]:.2f}%")
    
    csv_path = os.path.join(os.path.dirname(__file__), 'bit_length_analysis.csv')
    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([f"\n=== Overall Statistics ==="])
        writer.writerow([f"Total Success times: {total_success},  Total Fail times: {total_fail}  Overall Success Rate: {(total_success / total_tests * 100):.2f}%"])
        overall_stats_row = [
            f"{overall_avg_accuracy[0]:.2f}", f"{overall_avg_accuracy[1]:.2f}",
            f"{overall_avg_accuracy[2]:.2f}", f"{overall_avg_accuracy[3]:.2f}"
        ]
        writer.writerow(overall_stats_row)


if __name__ == '__main__':
    main()