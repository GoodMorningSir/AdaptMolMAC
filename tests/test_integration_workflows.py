"""Integration-style tests for example and CLI workflows."""

from __future__ import annotations

import importlib.util
import io
import json
import pathlib
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np

import AdaptMolMAC as amm
import AdaptMolMAC.cli as cli


GENERATOR = [[1, 1, 1], [1, 0, 1], [1, 0, 0]]
PREAMBLE = amm.generate_preamble_bits(amm.Settings.PREAMBLE_NUM)
CHECK_PREFIX = PREAMBLE + amm.Settings.CHECK_CODE


def load_basic_pipeline_module():
    """Load the example module directly from the repository path."""
    module_path = pathlib.Path(__file__).resolve().parents[1] / "examples" / "basic_pipeline.py"
    spec = importlib.util.spec_from_file_location("adaptmolmac_basic_pipeline", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class BasicPipelineIntegrationTests(unittest.TestCase):
    """Smoke-test the real example workflow under a fixed random seed."""

    def test_basic_pipeline_recovers_expected_payloads_with_fixed_seed(self):
        module = load_basic_pipeline_module()
        amm.Logger.disable()
        np.random.seed(0)

        output_buffer = io.StringIO()
        with redirect_stdout(output_buffer):
            module.main()

        output = output_buffer.getvalue()
        self.assertIn("Decoded channel count: 2", output)
        self.assertIn(
            "Recovered payloads after Viterbi: ['01010011010', '10010011011']",
            output,
        )


class CliRunMockEndToEndTests(unittest.TestCase):
    """Verify that `cli.run()` emits expected progress and marker output."""

    def test_run_prints_machine_readable_markers(self):
        encoded_1 = CHECK_PREFIX + "111000"
        encoded_2 = CHECK_PREFIX + "000111"
        decoded_1 = encoded_1 + "101"
        decoded_2 = encoded_2
        error_detector_calls = []

        class FakeSignal:
            def __init__(self, labels=None):
                self.labels = [] if labels is None else list(labels)

            def __add__(self, other):
                return FakeSignal(self.labels + other.labels)

        class FakeTx:
            def __init__(self, interval, tx_offset, amplitude, n_preamble, viterbi_gen):
                self.interval = interval
                self.tx_offset = tx_offset
                self.amplitude = amplitude
                self.n_preamble = n_preamble
                self.viterbi_gen = viterbi_gen
                self.encode_data = []

            def transmit(self, bits):
                encoded_payload = {
                    "10101": encoded_1,
                    "01010": encoded_2,
                }[bits]
                self.encode_data.append(encoded_payload)
                return FakeSignal([encoded_payload])

        class FakeSignalProcessor:
            def process_signal(self, y_rx, x_offset):
                return y_rx

            def kalman_filter(self, y_rx):
                return y_rx

            def adaptive_threshold_filter(self, y_rx):
                return y_rx

            def retain_peak_points_filter(self, peak_points_num, yrx_data):
                return yrx_data

        class FakeChannelInfo:
            def __init__(self):
                self.yRx = [0.0, 1.0, 0.0]
                self.reset_called = False

            def reset_start_pos(self):
                self.reset_called = True

        class FakeStationaryProcessor:
            def __init__(self, n_preamble):
                self.n_preamble = n_preamble
                self.preamble = PREAMBLE
                self.channel_info = FakeChannelInfo()

            def estimate_channel(self, y_rx, filtered_y_rx, interval_dev=0, start_pos_dev=0):
                return self.channel_info, 0.125

        class FakeDecoder:
            def __init__(self, y_rx, channels, generator):
                self.y_rx = y_rx
                self.channels = channels
                self.generator = generator

            def decode(self, MAX_SIGNAL_NUM=None):
                return [decoded_1, decoded_2]

        class FakeErrorDetect:
            def __init__(self, seed, *args):
                error_detector_calls.append((seed, args))

            def compare_accuracy(self):
                if len(error_detector_calls) == 1:
                    return [100.0, 50.0]
                return [80.0, 100.0]

        def fake_viterbi_decode(bits, generators, constraint_length):
            return {
                "111000": "10101",
                "000111": "01010",
            }[bits]

        fake_params = {
            "seed": 7,
            "generator": GENERATOR,
            "chan_para1": [17, 50, 1.0],
            "chan_para2": [22, 864, 1.0],
            "tran_data1": "10101",
            "tran_data2": "01010",
        }
        for i in range(3, cli.MAX_TRANSMITTERS + 1):
            fake_params[f"chan_para{i}"] = []
            fake_params[f"tran_data{i}"] = ""

        output_buffer = io.StringIO()
        with (
            patch.object(cli, "get_run_params", return_value=fake_params),
            patch.object(cli.time, "time", side_effect=[100.0, 100.5]),
            patch.object(cli.AdaptMolMAC, "ChannelModel_Tx", FakeTx),
            patch.object(cli.AdaptMolMAC, "yRxData", FakeSignal),
            patch.object(cli.AdaptMolMAC, "SignalProcessor", FakeSignalProcessor),
            patch.object(cli.AdaptMolMAC, "StationaryProcessor", FakeStationaryProcessor),
            patch.object(cli.AdaptMolMAC, "DynamicDecoder", FakeDecoder),
            patch.object(cli.AdaptMolMAC, "ErrorDetect", FakeErrorDetect),
            patch.object(cli.AdaptMolMAC, "viterbi_decode", side_effect=fake_viterbi_decode),
            redirect_stdout(output_buffer),
        ):
            cli.run()

        output = output_buffer.getvalue()
        self.assertIn("[Main] Loading runtime parameters", output)
        self.assertIn("[Main] Initializing transmitters", output)
        self.assertIn("[Main] Decoding channels", output)
        self.assertIn("ACC_RATES_START:[100.0, 50.0, 80.0, 100.0]:ACC_RATES_END", output)
        self.assertIn("TX_COUNT_START:2:TX_COUNT_END", output)
        self.assertIn("CHNL_PRED_COUNT_START:2:CHNL_PRED_COUNT_END", output)

        marker_payload = output.split("ACC_RATES_START:", 1)[1].split(":ACC_RATES_END", 1)[0]
        self.assertEqual(json.loads(marker_payload), [100.0, 50.0, 80.0, 100.0])
        self.assertEqual(len(error_detector_calls), 2)
        self.assertEqual(error_detector_calls[0][0], 7)


if __name__ == "__main__":
    unittest.main()
