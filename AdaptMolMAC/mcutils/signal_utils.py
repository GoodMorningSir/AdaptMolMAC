
"""Binary-string utility helpers for the AdaptMolMAC pipeline.

This module contains lightweight validation and conversion helpers together
with the preamble generator shared by simulation and decoding code.
"""

class BinaryUtils:
    """Validate and convert binary data used across the package.

    This helper class keeps binary-string validation and conversion logic in a
    single place so that simulation, coding, and decoding components use the
    same conventions.
    """

    @staticmethod
    def validate_binary_string(input_str):
        """Validate a binary string.

        Args:
            input_str (object): Candidate value to validate.

        Returns:
            bool: True if the value is a string made only of `0` and `1`.
        """
        if not isinstance(input_str, str):
            return False
        return all(char in '01' for char in input_str)

    @staticmethod
    def validate_binary_list(input_list):
        """Validate a list of binary integers.

        Args:
            input_list (object): Candidate value to validate.

        Returns:
            bool: True if the value is a list of integers containing only 0 and
            1.
        """
        if not isinstance(input_list, list):
            return False
        return all(isinstance(item, int) and item in (0, 1) for item in input_list)

    @staticmethod
    def binary_string_to_list(input_str):
        """Convert a binary string into a list of integer bits.

        Args:
            input_str (str): Input binary string.

        Returns:
            list[int]: Converted bit list.

        Raises:
            ValueError: If `input_str` is not a valid binary string.
        """
        if not BinaryUtils.validate_binary_string(input_str):
            raise ValueError("Input string is not a valid binary string.")
        return [int(char) for char in input_str]

    @staticmethod
    def list_to_binary_string(input_list):
        """Convert a list of integer bits into a binary string.

        Args:
            input_list (list[int]): Input bit list.

        Returns:
            str: Converted binary string.

        Raises:
            ValueError: If `input_list` is not a valid binary list.
        """
        if not BinaryUtils.validate_binary_list(input_list):
            raise ValueError("Input list is not a valid binary list.")
        return ''.join(str(item) for item in input_list)

def generate_preamble_bits(n):
        """Generate the triangular-spacing preamble used by the receiver.

        Args:
            n (int): Preamble order.

        Returns:
            str | list[int]: Generated preamble bit string. When `n < 1`, the
            function returns `[1]` for backward compatibility with existing
            callers.

        Raises:
            ValueError: If `n` is negative or not an integer.
        """
        if not isinstance(n, int) or n < 0:
            raise ValueError("n must be a natural number.")
        if n < 1:
            return [1]
        bits = [1, 1]
        for i in range(2, n + 1):
            bits += [0] * (i - 1) + [1]
        return BinaryUtils.list_to_binary_string(bits)
