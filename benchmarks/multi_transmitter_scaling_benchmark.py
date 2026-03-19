"""Benchmark decoding performance versus transmitter count.

This script launches the `AdaptMolMAC.cli` entry point many times for different
numbers of active transmitters, collects success and accuracy statistics, and
writes summary and error-bar CSV outputs.
"""

import subprocess
import random
import numpy as np
import json
import sys
import os
import csv
import time
from multiprocessing import Pool
import math
from pathlib import Path
MAX_WORKERS = 1


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
CLI_MODULE = "AdaptMolMAC.cli"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import AdaptMolMAC
import traceback

times_per_transmitter = 500
transmitter_counts = [1,2,3,4]
bit_length = 20

transmitter_stats = {}

intervals = [14, 17, 21, 24]
offsets = [50, 653, 1200, 1851]
amplitudes = [1.0, 1.0, 1.0, 1.0]


ACC_LAYOUT = "grouped"

def split_acc_rates(acc, t_count):
    """Split accuracy values into send and receive groups.

    Args:
        acc (list[float]): Flat accuracy list parsed from the CLI entry point.
        t_count (int): Number of active transmitters.

    Returns:
        tuple[list[float], list[float]]: Send-side and receive-side accuracy
        values.
    """
    if not acc:
        return [], []
    if len(acc) < 2 * t_count:
        return [], []

    if ACC_LAYOUT == "interleaved":
        send_vals = [acc[2*(i-1)] for i in range(1, t_count+1)]
        recv_vals = [acc[2*(i-1)+1] for i in range(1, t_count+1)]
    else:  # "grouped"
        send_vals = [acc[i-1] for i in range(1, t_count+1)]
        recv_vals = [acc[t_count + (i-1)] for i in range(1, t_count+1)]

    return send_vals, recv_vals

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
    """Create the transmitter-count summary CSV file."""
    csv_path = os.path.join(os.path.dirname(__file__), 'transmitter_count_analysis.csv')
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        headers = [
            'Transmitter_Count', 'Total_Tests', 'Success_Times', 'Fail_Times', 'Success_Rate(%)',
            'Avg_Execution_Time(ms)', 'Std_Execution_Time(ms)'
        ]
        for i in range(1, 5):
            headers.extend([f'Avg_tx{i}_send_accuracy(%)', ])
        for i in range(1, 5):
            headers.extend([f'Avg_tx{i}_recv_accuracy(%)'])
        writer.writerow(headers)
    
    print(f"CSV file created with headers: {csv_path}")

def write_result_row(transmitter_count, stats):
    """Append one transmitter-count summary row to the CSV file.

    Args:
        transmitter_count (int): Number of transmitters represented by the row.
        stats (dict): Aggregated statistics for that transmitter count.
    """
    csv_path = os.path.join(os.path.dirname(__file__), 'transmitter_count_analysis.csv')
    
    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        total_tests = stats['success_times'] + stats['fail_times']
        success_rate = (stats['success_times'] / total_tests * 100) if total_tests > 0 else 0
        
        avg_time = np.mean(stats['execution_times']) if stats['execution_times'] else 0
        std_time = np.std(stats['execution_times']) if stats['execution_times'] else 0
        
        row = [
            transmitter_count, total_tests, stats['success_times'], stats['fail_times'], 
            f"{success_rate:.2f}",
            f"{avg_time:.2f}", f"{std_time:.2f}"
        ]
        
        send_list = []
        recv_list = []
        for i in range(1, 5):
            if i <= transmitter_count and stats['accuracy_list']:
                send_accs, recv_accs = [], []
                for acc_list in stats['accuracy_list']:
                    svals, rvals = split_acc_rates(acc_list, transmitter_count)
                    if len(svals) >= i and len(rvals) >= i:
                        send_accs.append(svals[i-1])
                        recv_accs.append(rvals[i-1])
                if send_accs:
                    send_list.append(f"{np.mean(send_accs):.2f}")
                else:
                    send_list.append("0.00")
                if recv_accs:
                    recv_list.append(f"{np.mean(recv_accs):.2f}")
                else:
                    recv_list.append("0.00")
            else:
                send_list.append("N/A")
                recv_list.append("N/A")
        row.extend(send_list)
        row.extend(recv_list)
        writer.writerow(row)
        print(f"Result for transmitter count {transmitter_count} written to CSV")


def write_errorbar_summary(transmitter_stats, max_tx=4):
    """Write summary statistics with error bars for each transmitter count.

    Args:
        transmitter_stats (dict): Aggregated transmitter-count statistics.
        max_tx (int): Maximum number of transmitters to report.
    """
    csv_path = os.path.join(os.path.dirname(__file__), 'transmitter_count_errorbars.csv')
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        headers = [
            'Transmitter_Count',
            'Total_Tests',
            'Success_Rate(%)',
            'Success_Rate_Std(%)',
            'Avg_Execution_Time(ms)',
            'Std_Execution_Time(ms)'
        ]
        
        for i in range(1, max_tx + 1):
            headers.extend([f'tx{i}_send_mean(%)', f'tx{i}_send_std(%)'])
        for i in range(1, max_tx + 1):
            headers.extend([f'tx{i}_recv_mean(%)', f'tx{i}_recv_std(%)'])
        
        writer.writerow(headers)
        
        for t_count in sorted(transmitter_stats.keys()):
            stats = transmitter_stats[t_count]
            success_times = stats['success_times']
            fail_times = stats['fail_times']
            total_tests = success_times + fail_times
            
            if total_tests > 0:
                p = success_times / total_tests
                success_rate = p * 100.0
                success_rate_std = math.sqrt(p * (1.0 - p) / total_tests) * 100.0
            else:
                success_rate = 0.0
                success_rate_std = 0.0
            
            if stats['execution_times']:
                avg_time = float(np.mean(stats['execution_times']))
                std_time = float(np.std(stats['execution_times']))
            else:
                avg_time = 0.0
                std_time = 0.0
            
            row = [
                t_count,
                total_tests,
                f"{success_rate:.2f}",
                f"{success_rate_std:.2f}",
                f"{avg_time:.2f}",
                f"{std_time:.2f}"
            ]
            
            acc_list = stats['accuracy_list']
            
            for i in range(1, max_tx + 1):
                if i <= t_count and acc_list:
                    send_vals = []
                    recv_vals = []
                    for acc in acc_list:
                        svals, rvals = split_acc_rates(acc, t_count)
                        if len(svals) >= i and len(rvals) >= i:
                            send_vals.append(svals[i-1])
                            recv_vals.append(rvals[i-1])
                                            
                    if send_vals:
                        send_mean = float(np.mean(send_vals))
                        send_std = float(np.std(send_vals))
                        recv_mean = float(np.mean(recv_vals))
                        recv_std = float(np.std(recv_vals))
                        row.extend([
                            f"{send_mean:.2f}",
                            f"{send_std:.2f}",
                            f"{recv_mean:.2f}",
                            f"{recv_std:.2f}"
                        ])
                    else:
                        row.extend(["N/A", "N/A", "N/A", "N/A"])
                else:
                    row.extend(["N/A", "N/A", "N/A", "N/A"])
            
            writer.writerow(row)
    
    print(f"Errorbar summary CSV written to: {csv_path}")


def run_one_experiment(args):
    """Run one benchmark case in a worker process.

    Args:
        args (tuple): Packed experiment arguments.

    Returns:
        tuple[int, int, bool, float | None, list[float] | None]: Transmitter
        count, random seed, success flag, execution time in milliseconds, and
        parsed accuracy values.
    """
    (transmitter_count,
     seed,
     bit_length,
     intervals,
     offsets,
     amplitudes) = args

    generator = [
        [1, 1, 1],  # 0b111 (7)
        [1, 0, 1],  # 0b101 (5)
        [1, 0, 0]   # 0b110 (6)
    ]

    random.seed(seed)
    np.random.seed(seed)

    tran_data = {}
    chan_para = {}

    for j in range(1, transmitter_count + 1):
        tran_data[j] = random_bitstr(bit_length)
        interval = intervals[(j - 1) % len(intervals)]
        offset = offsets[(j - 1) % len(offsets)]
        amplitude = amplitudes[(j - 1) % len(amplitudes)]
        chan_para[j] = [interval, offset, amplitude]

    cmd_args = [sys.executable, "-m", CLI_MODULE,
                '--seed', str(seed),
                '--generator', json.dumps(generator)]

    for j in range(1, transmitter_count + 1):
        cmd_args.extend([f'--chan_para{j}', json.dumps(chan_para[j])])
        cmd_args.extend([f'--tran_data{j}', tran_data[j]])

    start_time = time.time()
    result = subprocess.run(
        cmd_args,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='ignore',
        cwd=str(PROJECT_ROOT),
    )
    end_time = time.time()
    execution_time = (end_time - start_time) * 1000.0  # ms
    
    success = ((result.returncode == 0))
    
    if not success:
        try:
            log_path = os.path.join(os.path.dirname(__file__), 'error_runs.log')
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            with open(log_path, 'a', encoding='utf-8') as lf:
                lf.write('=' * 80 + '\n')
                lf.write(f'ERROR RUN: {timestamp}\n')
                lf.write(f'Transmitter_Count: {transmitter_count}\n')
                lf.write(f'Seed: {seed}\n')
                lf.write(f'Bit_length: {bit_length}\n')
                lf.write('Chan_para: ' + json.dumps(chan_para, ensure_ascii=False) + '\n')
                lf.write('Tran_data: ' + json.dumps(tran_data, ensure_ascii=False) + '\n')
                lf.write('Cmd: ' + ' '.join(cmd_args) + '\n')
                lf.write(f'Returncode: {result.returncode}\n')
                lf.write('STDOUT:\n')
                lf.write(result.stdout + '\n')
                lf.write('STDERR:\n')
                lf.write(result.stderr + '\n')
                lf.write('=' * 80 + '\n\n')
        except Exception as e:
            try:
                AdaptMolMAC.logger.log(f"Failed to write error log: {e}")
            except Exception:
                pass
    
    acc_rates, tx_count, chnl_pred_count = parse_cli_output(result.stdout)
    
    if success:
        assert tx_count is not None, "tx_count is None"
        assert chnl_pred_count is not None, "chnl_pred_count is None"
        assert tx_count == transmitter_count, f"Expected tx_count {transmitter_count}, got {tx_count}"
        assert chnl_pred_count<= transmitter_count, f"Expected chnl_pred_count <= {transmitter_count}, got {chnl_pred_count}"
        
        while chnl_pred_count < transmitter_count:
            acc_rates.insert(chnl_pred_count, 50.0)
            acc_rates.append(50.0)
            chnl_pred_count += 1
        
        return (transmitter_count, seed, True, execution_time, acc_rates)
    else:
        return (transmitter_count, seed, False, None, None)


def main():
    """Run the multi-transmitter scaling benchmark."""
    total_iterations = len(transmitter_counts) * times_per_transmitter
    current_iteration = 0
    
    for transmitter_count in transmitter_counts:
        transmitter_stats[transmitter_count] = {
            'success_times': 0,
            'fail_times': 0,
            'accuracy_list': [],
            'execution_times': []
        }
    
    write_results_to_csv()

    for transmitter_count in transmitter_counts:
        print(f"\n=== Testing transmitter Count: {transmitter_count} ===")

        task_args = []
        for _ in range(times_per_transmitter):
            seed = random.randint(0, 2 ** 32 - 1)
            task_args.append((
                transmitter_count,
                seed,
                bit_length,
                intervals,
                offsets,
                amplitudes
            ))

        with Pool(processes=MAX_WORKERS) as pool:
            for (t_count, seed, success, execution_time, acc_rates) in pool.imap_unordered(
                run_one_experiment, task_args
            ):
                current_iteration += 1
                print(
                    f'Running {CLI_MODULE}, iteration {current_iteration}/{total_iterations}, '
                    f'transmitter_count: {t_count}, seed {seed}'
                )

                stats = transmitter_stats[t_count]

                if success:
                    stats['success_times'] += 1
                    if acc_rates:
                        stats['accuracy_list'].append(acc_rates)
                    stats['execution_times'].append(execution_time)
                else:
                    stats['fail_times'] += 1

        stats = transmitter_stats[transmitter_count]
        total_tests = stats['success_times'] + stats['fail_times']
        success_rate = (stats['success_times'] / total_tests * 100) if total_tests > 0 else 0

        print(f"\n--- transmitter Count {transmitter_count} Results ---")
        print(f"Success times: {stats['success_times']}")
        print(f"Fail times: {stats['fail_times']}")
        print(f"Success rate: {success_rate:.2f}%")

        if stats['execution_times']:
            avg_time = np.mean(stats['execution_times'])
            std_time = np.std(stats['execution_times'])
            print(f"Average execution time: {avg_time:.2f} ms")
            print(f"Standard deviation of execution time: {std_time:.2f} ms")

        if stats['accuracy_list']:
            print("Accuracy results:")
            for j in range(1, transmitter_count + 1):
                if len(stats['accuracy_list']) > 0:
                    accuracies = []
                    for acc_list in stats['accuracy_list']:
                        if len(acc_list) >= 2 * j:
                            send_acc = acc_list[2 * (j - 1)]
                            recv_acc = acc_list[2 * (j - 1) + 1]
                            accuracies.append([send_acc, recv_acc])

                    if accuracies:
                        avg_send = np.mean([acc[0] for acc in accuracies])
                        avg_recv = np.mean([acc[1] for acc in accuracies])
                        print(f"  tx{j}_send_accuracy: {avg_send:.2f}%")
                        print(f"  tx{j}_recv_accuracy: {avg_recv:.2f}%")

        write_result_row(transmitter_count, stats)

    
    total_success = sum(stats['success_times'] for stats in transmitter_stats.values())
    total_fail = sum(stats['fail_times'] for stats in transmitter_stats.values())
    total_tests = total_success + total_fail
    
    print(f"\n=== Overall Statistics ===")
    print(f"Total Success times: {total_success}")
    print(f"Total Fail times: {total_fail}")
    print(f"Overall Success Rate: {(total_success / total_tests * 100):.2f}%")
    
    all_execution_times = []
    for stats in transmitter_stats.values():
        if stats['execution_times']:
            all_execution_times.extend(stats['execution_times'])
    
    if all_execution_times:
        overall_avg_time = np.mean(all_execution_times)
        overall_std_time = np.std(all_execution_times)
        print(f"\n=== Overall Execution Time ===")
        print(f"Overall average execution time: {overall_avg_time:.2f} ms")
        print(f"Overall standard deviation: {overall_std_time:.2f} ms")
    
    csv_path = os.path.join(os.path.dirname(__file__), 'transmitter_count_analysis.csv')
    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([f"\n=== Overall Statistics ==="])
        writer.writerow([f"Total Success times: {total_success}, Total Fail times: {total_fail}, Overall Success Rate: {(total_success / total_tests * 100):.2f}%"])
        if all_execution_times:
            writer.writerow([f"Overall average execution time: {overall_avg_time:.2f} ms, Overall standard deviation: {overall_std_time:.2f} ms"])
    write_errorbar_summary(transmitter_stats, max_tx=4)


if __name__ == '__main__':
    main()
