"""Smoke tests for the public package interface."""

import unittest

import AdaptMolMAC as amm


class PackageImportTests(unittest.TestCase):
    """Verify that the package exposes its main public API."""

    def test_version_is_defined(self):
        """The package should define a public version string."""
        self.assertIsInstance(amm.__version__, str)
        self.assertTrue(amm.__version__)

    def test_expected_exports_exist(self):
        """Common top-level symbols should remain importable."""
        for symbol_name in (
            "ChannelModel_Tx",
            "DynamicDecoder",
            "Settings",
            "SignalProcessor",
            "StationaryProcessor",
            "generate_preamble_bits",
            "viterbi_decode",
            "yRxData",
        ):
            self.assertTrue(hasattr(amm, symbol_name), symbol_name)


if __name__ == "__main__":
    unittest.main()
