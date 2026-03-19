"""Regression tests for transmitter and stationary-processor behavior."""

import math
import unittest
from unittest.mock import patch

import numpy as np

import AdaptMolMAC as amm
import AdaptMolMAC.processing.stationary_detector as stationary_detector


GENERATOR = [[1, 1, 1], [1, 0, 1], [1, 0, 0]]


class ChannelModelTxRegressionTests(unittest.TestCase):
    """Verify stable transmitter-side framing behavior."""

    @patch("AdaptMolMAC.models.channel_model.MCModel.send", return_value=np.array([0.25, 0.5, 0.75]))
    def test_transmit_prefixes_preamble_and_tracks_history(self, send_mock):
        tx = amm.ChannelModel_Tx(
            interval=17,
            tx_offset=50,
            amplitude=1.0,
            n_preamble=amm.Settings.PREAMBLE_NUM,
            viterbi_gen=GENERATOR,
        )

        signal = tx.transmit("1011")
        expected_payload = amm.convolutional_encode("1011", GENERATOR, len(GENERATOR[0]))
        expected_encoded = amm.generate_preamble_bits(amm.Settings.PREAMBLE_NUM)
        expected_encoded += amm.Settings.CHECK_CODE + expected_payload

        send_mock.assert_called_once_with(expected_encoded, AddNoise=False)
        self.assertEqual(tx.transmit_data, ["1011"])
        self.assertEqual(tx.encode_data, [expected_encoded])
        self.assertEqual(signal.send_bits, [expected_encoded])
        self.assertEqual(tx.parameters, {"interval": 17, "tx_offset": 50})


class StationaryProcessorRegressionTests(unittest.TestCase):
    """Verify deterministic fitting helpers and failure handling."""

    def test_non_continuous_matching_recovers_exact_triangular_pattern(self):
        d, s, err = amm.StationaryProcessor.non_continuous_matching([5, 15, 35, 65], 4)

        self.assertAlmostEqual(float(d), 5.0)
        self.assertAlmostEqual(float(s), 10.0)
        self.assertLess(float(err), 1e-9)

    def test_estimate_channel_returns_none_when_peaks_are_insufficient(self):
        processor = amm.StationaryProcessor(n_preamble=amm.Settings.PREAMBLE_NUM)
        signal = amm.yRxData(process_data=np.zeros(80), ifLogger=False)
        sparse_points = stationary_detector.StationaryPoints([(5, "peak"), (15, "peak")])

        with patch.object(processor, "detect", return_value=sparse_points):
            channel, mse = processor.estimate_channel(signal, signal)

        self.assertIsNone(channel)
        self.assertTrue(math.isinf(mse))

    def test_estimate_channel_uses_detected_peaks_and_returns_fitted_channel(self):
        processor = amm.StationaryProcessor(n_preamble=amm.Settings.PREAMBLE_NUM)
        signal_values = np.zeros(90)
        signal_values[[5, 15, 35, 65]] = [1.0, 1.2, 1.4, 1.6]
        signal = amm.yRxData(process_data=signal_values, ifLogger=False)
        peak_points = stationary_detector.StationaryPoints(
            [(5, "peak"), (15, "peak"), (35, "peak"), (65, "peak")]
        )

        with (
            patch.object(processor, "detect", return_value=peak_points),
            patch.object(stationary_detector.ChanneModel_Rx, "train_keypoint", return_value=0.25),
            patch.object(stationary_detector.ChanneModel_Rx, "fine_tune_params", return_value=0.125),
        ):
            channel, mse = processor.estimate_channel(signal, signal)

        self.assertIsNotNone(channel)
        self.assertEqual(channel.interval, 10)
        self.assertEqual(channel.key_point1[0], 5)
        self.assertEqual(channel.preamble_bits, processor.preamble)
        self.assertEqual(mse, 0.125)


if __name__ == "__main__":
    unittest.main()
