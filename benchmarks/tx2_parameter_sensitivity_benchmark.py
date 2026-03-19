"""Benchmark sensitivity to transmitter-2 parameter perturbations.

This script evaluates one-at-a-time and grid-style perturbations applied to the
second transmitter while keeping the first transmitter fixed. Results are
written to summary and detailed CSV files.
"""

import subprocess
import random
import numpy as np
import json
import sys
import os
import csv
import itertools
from multiprocessing import Pool, cpu_count
from pathlib import Path

MAX_WORKERS = 1
TIMEOUT_SECONDS = 120
OFFSET_FIX_MARGIN = 14

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
CLI_MODULE = "AdaptMolMAC.cli"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import AdaptMolMAC
import traceback

times_per_test = 200
bit_length = 50

preamble_length = len(AdaptMolMAC.mcutils.generate_preamble_bits(AdaptMolMAC.Settings.PREAMBLE_NUM))

variation_factors = [
    0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0,
    1.0 / 0.95, 1.0 / 0.9, 1.0 / 0.85, 1.0 / 0.8, 1.0 / 0.7, 2.0
]

base_params1 = {'interval': 17, 'offset': 50,  'amplitude': 1.0}

base_params2 = {'interval': 22, 'offset': 864, 'amplitude': 1.0}

test_types = [
    {'name': 'interval_only',  'mode': 'oat',  'params': ['interval'],   'apply_offset_fix': False},
    {'name': 'offset_only',    'mode': 'oat',  'params': ['offset'],     'apply_offset_fix': False},
    {'name': 'amplitude_only', 'mode': 'oat',  'params': ['amplitude'],  'apply_offset_fix': False},

    {'name': 'interval_offset_grid',    'mode': 'grid', 'params': ['interval', 'offset'],     'apply_offset_fix': True},
    {'name': 'interval_amplitude_grid', 'mode': 'grid', 'params': ['interval', 'amplitude'],  'apply_offset_fix': True},
    {'name': 'offset_amplitude_grid',   'mode': 'grid', 'params': ['offset', 'amplitude'],    'apply_offset_fix': True},
]

def random_bitstr(length: int) -> str:
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
        lines = (output_text or "").strip().split('\n')
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
        try:
            AdaptMolMAC.logger.log(f"Error parsing output: {e}")
        except Exception:
            pass
        return None, None, None

def validate_offset_constraint(interval1, offset1, offset2) -> bool:
    """Validate the guarded offset relationship between the two transmitters.

    Args:
        interval1 (int): Transmitter-1 interval.
        offset1 (int): Transmitter-1 offset.
        offset2 (int): Candidate transmitter-2 offset.

    Returns:
        bool: True if the offset is valid.
    """
    min_offset2 = offset1 + int(preamble_length * 1.5) * interval1
    return offset2 > min_offset2

def apply_factors_to_tx2(base2, factors):
    """Apply multiplicative factors to transmitter-2 parameters only.

    Args:
        base2 (dict): Base transmitter-2 parameter dictionary.
        factors (dict): Multiplicative factors for interval, offset, and
            amplitude.

    Returns:
        dict: Perturbed transmitter-2 parameter dictionary.
    """
    fi = factors.get('interval', 1.0)
    fo = factors.get('offset', 1.0)
    fa = factors.get('amplitude', 1.0)

    interval2 = int(round(base2['interval'] * fi))
    interval2 = max(1, interval2)

    offset2 = int(round(base2['offset'] * fo))
    offset2 = max(0, offset2)

    amp2 = float(base2['amplitude'] * fa)

    return {'interval': interval2, 'offset': offset2, 'amplitude': amp2}

def maybe_fix_tx2_offset(test_type, params1_fixed, params2):
    """Repair transmitter-2 offset when a configuration violates spacing.

    Args:
        test_type (dict): Test configuration dictionary.
        params1_fixed (dict): Fixed transmitter-1 parameters.
        params2 (dict): Mutable transmitter-2 parameters.

    Returns:
        bool: True if the offset was adjusted, otherwise False.
    """
    if not test_type.get('apply_offset_fix', False):
        return False

    if 'offset' in test_type.get('params', []):
        return False

    if validate_offset_constraint(params1_fixed['interval'], params1_fixed['offset'], params2['offset']):
        return False

    min_offset2 = params1_fixed['offset'] + int(preamble_length * 1.5) * params1_fixed['interval']
    params2['offset'] = max(params2['offset'], min_offset2 + OFFSET_FIX_MARGIN)
    return True

def log_failure(test_type_name, seed, factors, chan_para1, chan_para2, tran_data1, tran_data2,
                cmd_list, reason, stdout_text="", stderr_text="", returncode=None):
    """Append one failed run to the shared failure log.

    Args:
        test_type_name (str): Name of the active test configuration.
        seed (int): Random seed used for the run.
        factors (dict): Applied perturbation factors.
        chan_para1 (str): Serialized transmitter-1 parameters.
        chan_para2 (str): Serialized transmitter-2 parameters.
        tran_data1 (str): Transmitter-1 payload.
        tran_data2 (str): Transmitter-2 payload.
        cmd_list (list[str]): Executed command.
        reason (str): Failure reason label.
        stdout_text (str): Captured standard output.
        stderr_text (str): Captured standard error.
        returncode (int | None): Process return code.
    """
    try:
        log_path = os.path.join(os.path.dirname(__file__), 'error_runs.log')
        with open(log_path, 'a', encoding='utf-8') as lf:
            lf.write('=' * 80 + '\n')
            lf.write(f"Reason: {reason}\n")
            lf.write(f"Test: {test_type_name}\n")
            lf.write(f"Seed: {seed}\n")
            lf.write(f"Factors_tx2: {factors}\n")
            lf.write(f"Returncode: {returncode}\n")
            lf.write('Chan_para1: ' + chan_para1 + '\n')
            lf.write('Chan_para2: ' + chan_para2 + '\n')
            lf.write('Tran_data1: ' + tran_data1 + '\n')
            lf.write('Tran_data2: ' + tran_data2 + '\n')
            lf.write('Cmd: ' + ' '.join(cmd_list) + '\n')
            lf.write('STDOUT:\n' + (stdout_text or "") + '\n')
            lf.write('STDERR:\n' + (stderr_text or "") + '\n')
            lf.write('=' * 80 + '\n\n')
    except Exception:
        pass

def write_results_to_csv():
    """Create the summary CSV file for the parameter-variation sweep."""
    csv_path = os.path.join(os.path.dirname(__file__), 'parameter_variation_analysis.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        headers = [
            'Test_Type',
            'Factor_interval_tx2', 'Factor_offset_tx2', 'Factor_amplitude_tx2',
            'Total_Tests', 'Success_Times', 'Fail_Times', 'Success_Rate(%)',
            'Avg_tx1_send_accuracy(%)', 'Std_tx1_send_accuracy(%)',
            'Avg_tx2_send_accuracy(%)', 'Std_tx2_send_accuracy(%)',
            'Avg_tx1_recv_accuracy(%)', 'Std_tx1_recv_accuracy(%)',
            'Avg_tx2_recv_accuracy(%)', 'Std_tx2_recv_accuracy(%)',
            'Interval1', 'Offset1', 'Amplitude1',
            'Interval2', 'Offset2', 'Amplitude2',
        ]
        w.writerow(headers)
    print(f"Summary CSV created: {csv_path}")

def write_result_row(test_name, factors, stats, params1, params2):
    """Append one summary row to the parameter-variation CSV file.

    Args:
        test_name (str): Name of the active test configuration.
        factors (dict): Applied perturbation factors.
        stats (dict): Aggregated success and accuracy statistics.
        params1 (dict): Effective transmitter-1 parameters.
        params2 (dict): Effective transmitter-2 parameters.
    """
    csv_path = os.path.join(os.path.dirname(__file__), 'parameter_variation_analysis.csv')
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)

        total_tests = stats['success_times'] + stats['fail_times']
        success_rate = (stats['success_times'] / total_tests * 100.0) if total_tests else 0.0

        avg_tx1_send = avg_tx2_send = avg_tx1_recv = avg_tx2_recv = 0.0
        std_tx1_send = std_tx2_send = std_tx1_recv = std_tx2_recv = 0.0

        if stats['accuracy_list']:
            a = np.array(stats['accuracy_list'], dtype=float)
            avg_tx1_send = float(np.mean(a[:, 0]))
            avg_tx2_send = float(np.mean(a[:, 1]))
            avg_tx1_recv = float(np.mean(a[:, 2]))
            avg_tx2_recv = float(np.mean(a[:, 3]))
            if len(a) > 1:
                std_tx1_send = float(np.std(a[:, 0], ddof=1))
                std_tx2_send = float(np.std(a[:, 1], ddof=1))
                std_tx1_recv = float(np.std(a[:, 2], ddof=1))
                std_tx2_recv = float(np.std(a[:, 3], ddof=1))

        row = [
            test_name,
            factors.get('interval', 1.0), factors.get('offset', 1.0), factors.get('amplitude', 1.0),
            total_tests, stats['success_times'], stats['fail_times'], success_rate,
            avg_tx1_send, std_tx1_send,
            avg_tx2_send, std_tx2_send,
            avg_tx1_recv, std_tx1_recv,
            avg_tx2_recv, std_tx2_recv,
            params1['interval'], params1['offset'], params1['amplitude'],
            params2['interval'], params2['offset'], params2['amplitude'],
        ]
        w.writerow(row)

def init_detailed_results_csv():
    """Create the detailed per-seed CSV file."""
    csv_path = os.path.join(os.path.dirname(__file__), 'parameter_variation_detail.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        headers = [
            'Test_Type',
            'Factor_interval_tx2', 'Factor_offset_tx2', 'Factor_amplitude_tx2',
            'Seed', 'Success',
            'tx1_send_accuracy(%)', 'tx2_send_accuracy(%)',
            'tx1_recv_accuracy(%)', 'tx2_recv_accuracy(%)'
        ]
        w.writerow(headers)
    print(f"Detail CSV created: {csv_path}")

def write_detailed_rows(test_name, factors, seeds, results):
    """Append per-seed results to the detailed CSV file.

    Args:
        test_name (str): Name of the active test configuration.
        factors (dict): Applied perturbation factors.
        seeds (list[int]): Random seeds used for the runs.
        results (list[dict]): Worker results aligned with `seeds`.
    """
    csv_path = os.path.join(os.path.dirname(__file__), 'parameter_variation_detail.csv')
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        for seed, res in zip(seeds, results):
            if res['success'] and res['acc_rates'] is not None:
                a = res['acc_rates']
                row = [
                    test_name,
                    factors.get('interval', 1.0), factors.get('offset', 1.0), factors.get('amplitude', 1.0),
                    seed, 1,
                    a[0], a[1], a[2], a[3]
                ]
            else:
                row = [
                    test_name,
                    factors.get('interval', 1.0), factors.get('offset', 1.0), factors.get('amplitude', 1.0),
                    seed, 0,
                    '', '', '', ''
                ]
            w.writerow(row)

def build_test_configs():
    """Build all one-at-a-time and grid test configurations.

    Returns:
        list[tuple[dict, dict]]: Test-type dictionary paired with factor
        dictionary.
    """
    configs = []
    for tt in test_types:
        if tt['mode'] == 'oat':
            p = tt['params'][0]
            for f in variation_factors:
                factors = {'interval': 1.0, 'offset': 1.0, 'amplitude': 1.0}
                factors[p] = f
                configs.append((tt, factors))
        elif tt['mode'] == 'grid':
            p1, p2 = tt['params']
            for f1, f2 in itertools.product(variation_factors, variation_factors):
                factors = {'interval': 1.0, 'offset': 1.0, 'amplitude': 1.0}
                factors[p1] = f1
                factors[p2] = f2
                configs.append((tt, factors))
        else:
            raise ValueError(f"Unknown mode: {tt['mode']}")
    return configs

def run_single_case(args):
    """Run one parameter-variation case for a single random seed.

    Args:
        args (tuple): Packed worker arguments `(test_type, factors, seed)`.

    Returns:
        dict: Worker result containing success status and parsed accuracy data.
    """
    test_type, factors, seed = args

    try:
        random.seed(seed)
        np.random.seed(seed)

        params1 = dict(base_params1)

        params2 = apply_factors_to_tx2(base_params2, factors)

        maybe_fix_tx2_offset(test_type, params1, params2)

        generator = [
            [1, 1, 1],
            [1, 0, 1],
            [1, 0, 0]
        ]
        tran_data1 = random_bitstr(bit_length)
        tran_data2 = random_bitstr(bit_length)

        chan_para1 = json.dumps([params1['interval'], params1['offset'], params1['amplitude']])
        chan_para2 = json.dumps([params2['interval'], params2['offset'], params2['amplitude']])

        args_cmd = [
            sys.executable, "-m", CLI_MODULE,
            '--seed', str(seed),
            '--generator', json.dumps(generator),
            '--chan_para1', chan_para1,
            '--chan_para2', chan_para2,
            '--tran_data1', tran_data1,
            '--tran_data2', tran_data2
        ]

        try:
            result = subprocess.run(
                args_cmd,
                capture_output=True,
                text=True, encoding='utf-8', errors='ignore',
                timeout=TIMEOUT_SECONDS,
                cwd=str(PROJECT_ROOT),
            )
        except subprocess.TimeoutExpired as te:
            log_failure(
                test_type_name=test_type['name'],
                seed=seed,
                factors=factors,
                chan_para1=chan_para1,
                chan_para2=chan_para2,
                tran_data1=tran_data1,
                tran_data2=tran_data2,
                cmd_list=args_cmd,
                reason=f"TIMEOUT>{TIMEOUT_SECONDS}s",
                stdout_text=getattr(te, "stdout", "") or "",
                stderr_text=getattr(te, "stderr", "") or "",
                returncode=None
            )
            return {'success': False, 'acc_rates': None}

        success = (result.returncode == 0)

        if not success:
            log_failure(
                test_type_name=test_type['name'],
                seed=seed,
                factors=factors,
                chan_para1=chan_para1,
                chan_para2=chan_para2,
                tran_data1=tran_data1,
                tran_data2=tran_data2,
                cmd_list=args_cmd,
                reason="NONZERO_RETURN",
                stdout_text=result.stdout,
                stderr_text=result.stderr,
                returncode=result.returncode
            )

        acc_rates, tx_count, chnl_pred_count = parse_cli_output(result.stdout)

        if success and acc_rates is not None and chnl_pred_count is not None:
            while chnl_pred_count < 2:
                acc_rates.insert(chnl_pred_count, 50.0)
                acc_rates.append(50.0)
                chnl_pred_count += 1

        return {
            'success': success,
            'acc_rates': acc_rates if (success and acc_rates) else None
        }

    except Exception:
        try:
            AdaptMolMAC.logger.log("Error in worker:\n" + traceback.format_exc())
        except Exception:
            pass
        return {'success': False, 'acc_rates': None}

def main():
    """Run the transmitter-2 parameter sensitivity benchmark."""
    AdaptMolMAC.config.Settings.DEFAULT_EXTENDED_BITS_NUM = 4
    print(f"Preamble length (bits): {preamble_length}")

    num_workers = min(MAX_WORKERS, cpu_count())
    print(f"Using {num_workers} worker processes")

    write_results_to_csv()
    init_detailed_results_csv()

    configs = build_test_configs()
    print(f"Total configurations: {len(configs)}")
    print(f"Total runs (config * times_per_test): {len(configs) * times_per_test}")

    test_stats = {}

    with Pool(processes=num_workers) as pool:
        for idx, (test_type, factors) in enumerate(configs, 1):
            test_name = test_type['name']
            key = (test_name, factors['interval'], factors['offset'], factors['amplitude'])

            if key not in test_stats:
                test_stats[key] = {'success_times': 0, 'fail_times': 0, 'accuracy_list': []}

            print(f"\n[{idx}/{len(configs)}] === {test_name} factors_tx2={factors} ===")

            seeds = [random.randint(0, 2**32 - 1) for _ in range(times_per_test)]
            tasks = [(test_type, factors, s) for s in seeds]

            results = pool.map(run_single_case, tasks)

            write_detailed_rows(test_name, factors, seeds, results)

            stats = test_stats[key]
            for res in results:
                if res['success']:
                    stats['success_times'] += 1
                    if res['acc_rates'] is not None:
                        stats['accuracy_list'].append(res['acc_rates'])
                else:
                    stats['fail_times'] += 1

            total_tests = stats['success_times'] + stats['fail_times']
            success_rate = (stats['success_times'] / total_tests * 100.0) if total_tests else 0.0
            print(f"Success: {stats['success_times']}, Fail: {stats['fail_times']}, Rate: {success_rate:.2f}%")

            params1 = dict(base_params1)
            params2 = apply_factors_to_tx2(base_params2, factors)
            maybe_fix_tx2_offset(test_type, params1, params2)

            write_result_row(test_name, factors, stats, params1, params2)

    total_success = sum(v['success_times'] for v in test_stats.values())
    total_fail = sum(v['fail_times'] for v in test_stats.values())
    total_tests = total_success + total_fail
    overall_success_rate = (total_success / total_tests * 100.0) if total_tests else 0.0

    print("\n=== Overall Statistics ===")
    print(f"Total Success times: {total_success}")
    print(f"Total Fail times: {total_fail}")
    print(f"Overall Success Rate: {overall_success_rate:.2f}%")

    csv_path = os.path.join(os.path.dirname(__file__), 'parameter_variation_analysis.csv')
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow([])
        w.writerow(["=== Overall Statistics ==="])
        w.writerow([f"Total Success times: {total_success}",
                    f"Total Fail times: {total_fail}",
                    f"Overall Success Rate: {overall_success_rate:.2f}%"])

    print(f"\nAll results have been written to: {csv_path}")


if __name__ == '__main__':
    main()
