"""Codebook generation utilities for small binary symbol sets.

The module builds compact codebooks while enforcing a minimum Hamming distance
between valid codewords.
"""

import random

class CodebookGenerator:
    """Build a small codebook with a minimum Hamming distance constraint.

    Attributes:
        capacity (int): Requested number of valid codewords.
        n_bits (int): Codeword length selected for the codebook.
        codebook (list[str]): Generated codewords.
        id_code_map (dict[int, str]): Mapping from identifier to codeword.
        code_id_map (dict[str, int]): Reverse mapping from codeword to
            identifier.
    """

    def __init__(self, capacity):
        """Create a codebook generator for the requested number of codewords.

        Args:
            capacity (int): Number of codewords that must be generated.
        """
        self.capacity = capacity
        self.n_bits = self._compute_min_n_bits()
        self.codebook = []
        self._generate_codebook()
        self.id_code_map = {i: code for i, code in enumerate(self.codebook)}
        self.code_id_map = {code: i for i, code in enumerate(self.codebook)}
    
    def _compute_min_n_bits(self):
        """Find the smallest codeword length that can hold the target capacity.

        Returns:
            int: Minimum candidate codeword length.
        """
        n_bits = 1
        while True:
            if (2 ** n_bits) / (1 + n_bits) >= self.capacity:
                return n_bits
            n_bits += 1
    
    def _hamming_distance(self, s1, s2):
        """Return the Hamming distance between two equal-length codewords.

        Args:
            s1 (str): First codeword.
            s2 (str): Second codeword.

        Returns:
            int: Hamming distance between the codewords.

        Raises:
            ValueError: If the codewords have different lengths.
        """
        if len(s1) != len(s2):
            raise ValueError("Codewords must have the same length.")
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))
    
    def _generate_codebook(self):
        """Populate the internal codebook using randomized search.

        The method repeatedly shuffles the candidate pool and greedily accepts
        codewords that satisfy the minimum Hamming-distance constraint. If the
        requested capacity cannot be reached, the codeword length is increased
        and the search restarts.
        """
        total_possible = 2 ** self.n_bits
        all_codes = [format(i, f'0{self.n_bits}b') for i in range(total_possible)]
        
        for _ in range(100):
            random.shuffle(all_codes)
            codebook = []
            for code in all_codes:
                if all(self._hamming_distance(code, existing) >= 3 for existing in codebook):
                    codebook.append(code)
                    if len(codebook) == self.capacity:
                        self.codebook = codebook
                        return
        
        self.n_bits += 1
        self._generate_codebook()
    
    def validate(self, code):
        """Check whether a codeword belongs to the generated codebook.

        Args:
            code (str): Candidate codeword.

        Returns:
            bool: True if the codeword is part of the generated codebook.
        """
        return code in self.codebook
    
    def get_code(self, id):
        """Return the codeword assigned to a given integer identifier.

        Args:
            id (int): Codeword identifier.

        Returns:
            str: Codeword mapped to the identifier.

        Raises:
            ValueError: If the identifier is outside the valid range.
        """
        if id < 0 or id >= self.capacity:
            raise ValueError(f"ID must be between 0 and {self.capacity - 1}.")
        return self.id_code_map[id]
    
    def get_id(self, code):
        """Return the identifier mapped to a codeword, if it exists.

        Args:
            code (str): Codeword to look up.

        Returns:
            int | None: Identifier for the codeword, or None when absent.
        """
        return self.code_id_map.get(code, None)
    
    def __str__(self):
        """Render the generated codebook as a human-readable string.

        Returns:
            str: One codeword per line prefixed by its identifier.
        """
        return '\n'.join(f"{id}: {code}" for id, code in enumerate(self.codebook))


if __name__ == "__main__":
    generator = CodebookGenerator(4)
    
    print(f"[CodebookGenerator] Generated codebook ({generator.n_bits} bits):")
    print(f"[CodebookGenerator]\n{generator}")
    
    test_code = generator.codebook[0]
    print(f"[CodebookGenerator] Validate codeword '{test_code}': {generator.validate(test_code)}")
    
    test_id = 2
    print(f"[CodebookGenerator] Codeword for ID {test_id}: {generator.get_code(test_id)}")
    
    test_code = generator.codebook[3]
    print(f"[CodebookGenerator] ID for codeword '{test_code}': {generator.get_id(test_code)}")
