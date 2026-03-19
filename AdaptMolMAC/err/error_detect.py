"""Utilities for comparing transmitted and decoded bit sequences.

This module computes accuracy values and appends experiment summaries to a CSV
file located next to the current entry script.
"""

from ..mcutils import logger
import csv
import sys
import os

class ErrorDetect:
    """Compare transmitted and recovered bit strings.

    Attributes:
        Seed (str): Experiment seed or run identifier.
        pairs (list[tuple[str, str]]): Paired transmitted and decoded bit
            strings.
    """

    def __init__(self, Seed, *args):
        """Store paired transmit/receive bit sequences for one experiment.

        Args:
            Seed (str | int): Experiment seed or run identifier.
            *args: Alternating transmitted and decoded bit sequences.
        """
        self.Seed = self._to_str(Seed)
        self.pairs = []
        
        for i in range(0, len(args), 2):
            if i + 1 < len(args):
                send_bits = self._to_str(args[i])
                recv_bits = self._to_str(args[i+1])
                self.pairs.append((send_bits, recv_bits))

    def _to_str(self, bits):
        """Normalize supported bit containers into a plain string."""
        if isinstance(bits, list):
            if len(bits) == 1 and isinstance(bits[0], str):
                return bits[0]
            return ''.join(str(b) for b in bits)
        return str(bits)

    def _calc_acc(self, send_bits, recv_bits):
        """Compute bit-wise accuracy for one transmit/receive pair."""
        if len(send_bits) == 0 or len(recv_bits) < len(send_bits):
            return 0.0
        correct_num = sum([send_bits[i] == recv_bits[i] for i in range(len(send_bits))])
        logger.log(
            f"[ErrorDetect] Compared {len(send_bits)} transmitted bits; "
            f"correct_bits={correct_num}"
        )
        return (correct_num / len(send_bits)) * 100

    def compare_accuracy(self):
        """Evaluate all pairs and append the results to `result.csv`.

        Returns:
            list[float]: Accuracy percentage for each bit-sequence pair.
        """
        acc_rates = []
        send_bits_list = []
        recv_bits_list = []
        
        for i, (send_bits, recv_bits) in enumerate(self.pairs):
            acc_rate = self._calc_acc(send_bits, recv_bits)
            acc_rates.append(acc_rate)
            send_bits_list.append(send_bits)
            recv_bits_list.append(recv_bits)
            logger.log(f"[ErrorDetect] Pair {i + 1} accuracy={acc_rate:.2f}%")
        
        csv_row = [self.Seed]
        csv_row.extend([f'{rate:.2f}%' for rate in acc_rates])
        csv_row.extend(send_bits_list)
        csv_row.extend(recv_bits_list)
        
        headers = ['Seed']
        headers.extend([f'Acc{i+1}(%)' for i in range(len(self.pairs))])
        headers.extend([f'SendBits{i+1}' for i in range(len(self.pairs))])
        headers.extend([f'RecvBits{i+1}' for i in range(len(self.pairs))])
        
        main_file_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        csv_path = os.path.join(main_file_dir, 'result.csv')
        file_exists = os.path.isfile(csv_path)
        
        with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            if not file_exists:
                writer.writerow(headers)
            
            writer.writerow(csv_row)
        return acc_rates
