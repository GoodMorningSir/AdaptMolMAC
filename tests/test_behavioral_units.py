"""Behavioral unit tests for deterministic core helpers."""

import io
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import AdaptMolMAC as amm
from AdaptMolMAC.cli import get_run_params, print_stage, random_bitstr
from AdaptMolMAC.mcutils.signal_utils import BinaryUtils
from AdaptMolMAC.viterbi.viterbi import validate_viterbi_generator


GENERATOR = [[1, 1, 1], [1, 0, 1], [1, 0, 0]]


class BinaryUtilityBehaviorTests(unittest.TestCase):
    """Verify binary validation, conversion, and preamble generation."""

    def test_binary_validation_accepts_and_rejects_expected_inputs(self):
        self.assertTrue(BinaryUtils.validate_binary_string("101001"))
        self.assertFalse(BinaryUtils.validate_binary_string("102001"))
        self.assertFalse(BinaryUtils.validate_binary_string(101001))

        self.assertTrue(BinaryUtils.validate_binary_list([1, 0, 1, 1]))
        self.assertFalse(BinaryUtils.validate_binary_list([1, 2, 1]))
        self.assertFalse(BinaryUtils.validate_binary_list("1011"))

    def test_binary_conversions_round_trip(self):
        bits = "101001"
        self.assertEqual(BinaryUtils.binary_string_to_list(bits), [1, 0, 1, 0, 0, 1])
        self.assertEqual(BinaryUtils.list_to_binary_string([1, 0, 1, 0, 0, 1]), bits)

    def test_binary_conversions_reject_invalid_inputs(self):
        with self.assertRaises(ValueError):
            BinaryUtils.binary_string_to_list("10a1")

        with self.assertRaises(ValueError):
            BinaryUtils.list_to_binary_string([1, 0, 2, 1])

    def test_generate_preamble_bits_handles_edge_cases(self):
        self.assertEqual(amm.generate_preamble_bits(0), [1])
        self.assertEqual(amm.generate_preamble_bits(3), "1101001")

        with self.assertRaises(ValueError):
            amm.generate_preamble_bits(-1)


class ViterbiBehaviorTests(unittest.TestCase):
    """Verify generator validation and encoder/decoder compatibility."""

    def test_validate_viterbi_generator_returns_constraint_length(self):
        self.assertEqual(validate_viterbi_generator(GENERATOR), 3)

    def test_validate_viterbi_generator_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            validate_viterbi_generator([[1, 0], [1, 0, 1]])

        with self.assertRaises(ValueError):
            validate_viterbi_generator([[1, 0, 2], [1, 1, 0]])

    def test_viterbi_round_trip_recovers_original_payload(self):
        for payload in ("1", "1011", "01010011010"):
            with self.subTest(payload=payload):
                encoded = amm.convolutional_encode(payload, GENERATOR, len(GENERATOR[0]))
                decoded = amm.viterbi_decode(encoded, GENERATOR, len(GENERATOR[0]))
                self.assertEqual(decoded, payload)


class SignalContainerBehaviorTests(unittest.TestCase):
    """Verify `yRxData` merging and locking behavior."""

    def test_add_combines_raw_signals_and_send_bits(self):
        left = amm.yRxData(data=[1.0, 2.0], send_bits=["101"], ifLogger=False)
        right = amm.yRxData(data=[0.5, 0.5, 0.5], send_bits=["010"], ifLogger=False)

        merged = left + right

        self.assertEqual(merged.raw_data.tolist(), [1.5, 2.5, 0.5])
        self.assertEqual(merged.send_bits, ["101", "010"])

    def test_locked_container_rejects_raw_data_access(self):
        signal = amm.yRxData(process_data=[0.1, 0.2, 0.3], ifLogger=False)

        self.assertTrue(signal.is_locked)
        self.assertEqual(signal.yRxData.tolist(), [0.1, 0.2, 0.3])

        with self.assertRaises(AttributeError):
            _ = signal.raw_data

    def test_lock_releases_raw_buffer(self):
        signal = amm.yRxData(data=[0.1, 0.2, 0.3], ifLogger=False)

        with patch("AdaptMolMAC.models.channel_model.noiseParam.AddNoise", side_effect=lambda arr: arr):
            signal.lock()

        self.assertTrue(signal.is_locked)
        self.assertEqual(signal.yRxData.tolist(), [0.1, 0.2, 0.3])
        with self.assertRaises(AttributeError):
            _ = signal.raw_data


class CliHelperBehaviorTests(unittest.TestCase):
    """Verify lightweight CLI helpers without running the full pipeline."""

    def test_get_run_params_uses_defaults(self):
        with patch.object(sys, "argv", ["adaptmolmac"]):
            params = get_run_params()

        self.assertEqual(params["seed"], 0)
        self.assertEqual(params["generator"], [])
        self.assertEqual(params["chan_para1"], [])
        self.assertEqual(params["tran_data1"], "")

    def test_get_run_params_parses_json_and_payload_arguments(self):
        argv = [
            "adaptmolmac",
            "--seed",
            "7",
            "--generator",
            "[[1,1,1],[1,0,1],[1,0,0]]",
            "--chan_para1",
            "[17,50,1.0]",
            "--tran_data1",
            "10101",
        ]
        with patch.object(sys, "argv", argv):
            params = get_run_params()

        self.assertEqual(params["seed"], 7)
        self.assertEqual(params["generator"], GENERATOR)
        self.assertEqual(params["chan_para1"], [17, 50, 1.0])
        self.assertEqual(params["tran_data1"], "10101")

    def test_random_bitstr_returns_binary_string_with_requested_length(self):
        bitstr = random_bitstr(32)

        self.assertEqual(len(bitstr), 32)
        self.assertTrue(set(bitstr).issubset({"0", "1"}))

    def test_print_stage_uses_standardized_prefix(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            print_stage("Initializing transmitters")

        self.assertEqual(buffer.getvalue().strip(), "[Main] Initializing transmitters")


if __name__ == "__main__":
    unittest.main()
