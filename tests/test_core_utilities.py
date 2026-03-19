"""Lightweight behavioral tests for core utilities."""

import unittest

import AdaptMolMAC as amm


class CoreUtilityTests(unittest.TestCase):
    """Exercise small, deterministic helpers without long simulations."""

    def test_generate_preamble_bits_returns_binary_string(self):
        """Generated preamble bits should be a non-empty binary string."""
        preamble = amm.generate_preamble_bits(amm.Settings.PREAMBLE_NUM)
        self.assertIsInstance(preamble, str)
        self.assertTrue(preamble)
        self.assertTrue(set(preamble).issubset({"0", "1"}))

    def test_convolutional_encode_preserves_binary_output(self):
        """Convolutional encoding should return a binary string."""
        generators = [[1, 1, 1], [1, 0, 1], [1, 0, 0]]
        encoded = amm.convolutional_encode("1011", generators, len(generators[0]))
        self.assertIsInstance(encoded, str)
        self.assertTrue(encoded)
        self.assertTrue(set(encoded).issubset({"0", "1"}))

    def test_received_signal_container_initializes(self):
        """`yRxData` should expose the received-data buffer attribute."""
        signal = amm.yRxData()
        self.assertTrue(hasattr(signal, "yRxData"))


if __name__ == "__main__":
    unittest.main()
