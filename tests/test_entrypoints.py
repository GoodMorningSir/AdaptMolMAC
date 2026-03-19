"""Smoke tests for repository entry points."""

import importlib
import unittest


class EntrypointTests(unittest.TestCase):
    """Check that lightweight entry points import successfully."""

    def test_main_module_imports(self):
        """The repository-root wrapper should expose the packaged CLI module."""
        module = importlib.import_module("main")
        self.assertTrue(hasattr(module, "cli"))
        self.assertTrue(callable(module.cli.run))

    def test_cli_module_imports(self):
        """The packaged CLI implementation should be importable."""
        module = importlib.import_module("AdaptMolMAC.cli")
        self.assertTrue(callable(module.run))


if __name__ == "__main__":
    unittest.main()
