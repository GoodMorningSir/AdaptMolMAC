"""Convolutional coding and Viterbi decoding helpers.

This module contains the encoder, the classical Viterbi decoder, visualization
helpers, and an extended hypothesis decoder used by the dynamic multi-signal
pipeline.
"""

import itertools
import random
import itertools
from ..mcutils import BinaryUtils, logger
from ..config import Settings

PATH_DRAW = False

def validate_viterbi_generator(viterbi_gen):
    """Validate a convolutional-code generator matrix.

    Args:
        viterbi_gen (list[list[int]]): Candidate generator matrix.

    Returns:
        int: Constraint length inferred from the generator matrix.

    Raises:
        ValueError: If the matrix is not binary.
        ValueError: If rows do not all have the same length.
    """
    if not all(isinstance(row, list) and all(bit in (0, 1) for bit in row) for row in viterbi_gen):
        raise ValueError("viterbi_gen must be a 0-1 matrix")
    
    if len({len(row) for row in viterbi_gen}) != 1:
        raise ValueError("All rows in viterbi_gen must have the same length")
    
    constraint_length = len(viterbi_gen[0])
    return constraint_length

def convolutional_encode(input_bits, generators, constraint_length):
    """Encode a binary string with a convolutional encoder.

    Args:
        input_bits (str): Payload bits to encode.
        generators (list[list[int]]): Generator matrix.
        constraint_length (int): Encoder constraint length.

    Returns:
        str: Encoded binary string with termination bits appended.
    """
    input_bits = BinaryUtils.binary_string_to_list(input_bits)
    m = constraint_length
    input_bits = input_bits + [0]*(m-1)
    shift_register = [0]*(m-1)
    encoded = []
    
    for bit in input_bits:
        outputs = []
        for gen in generators:
            xor = 0
            for i in range(len(gen)):
                if i == 0:
                    xor ^= bit * gen[i]
                else:
                    xor ^= shift_register[i-1] * gen[i]
            outputs.append(xor % 2)
        
        encoded.extend(outputs)
        shift_register = [bit] + shift_register[:-1]
    
    encoded = BinaryUtils.list_to_binary_string(encoded)
    return encoded

def viterbi_decode(encoded, generators, constraint_length):
    """Decode a convolutionally encoded bit string.

    Args:
        encoded (str): Encoded bit string.
        generators (list[list[int]]): Generator matrix.
        constraint_length (int): Decoder constraint length.

    Returns:
        str: Decoded payload bit string.
    """
    encoded = BinaryUtils.binary_string_to_list(encoded)
    m = constraint_length
    num_states = 2 ** (m-1)
    state_pairs = list(itertools.product([0, 1], repeat=m-1))
    
    
    path_metrics = {s: float('inf') for s in state_pairs}
    path_metrics[tuple([0]*(m-1))] = 0
    
    path_history = [{} for _ in range(len(encoded)//3 + 1)]
    path_history[0] = {s: (None, None) for s in state_pairs}
    
    if PATH_DRAW:
        path_metrics_history = []
        path_metrics_history.append(path_metrics.copy())
    
    for t in range(len(encoded)//3):
        received = encoded[t*3 : (t+1)*3]
        new_metrics = {s: float('inf') for s in state_pairs}
        
        for prev_state in state_pairs:
            if path_metrics[prev_state] == float('inf'):
                continue
            
            for input_bit in [0, 1]:
                next_state = (input_bit,) + prev_state[:-1]
                outputs = []
                for gen in generators:
                    xor = 0
                    for i in range(len(gen)):
                        if i == 0:
                            xor ^= input_bit * gen[i]
                        else:
                            xor ^= prev_state[i-1] * gen[i]
                    outputs.append(xor % 2)
                
                metric = sum(r != o for r, o in zip(received, outputs))
                total_metric = path_metrics[prev_state] + metric
                
                if total_metric < new_metrics[next_state]:
                    new_metrics[next_state] = total_metric
                    path_history[t+1][next_state] = (prev_state, input_bit)
        path_metrics = new_metrics
        
        if PATH_DRAW:
            path_metrics_history.append(path_metrics.copy())
    
    if PATH_DRAW:
        fig = visualize_path_metrics(path_metrics_history, path_history)
        logger.log_image(fig, "Viterbi Path Metrics", "viterbi")
        

    current_state = tuple([0]*(m-1))
    decoded_bits = []
    if current_state not in path_history[len(encoded)//3]:
        current_state = min(path_metrics, key=path_metrics.get)
    
    for t in range(len(encoded)//3, 0, -1):
        
        prev_state, bit = path_history[t][current_state]
        decoded_bits.append(bit)
        current_state = prev_state
    
    decoded_bits = decoded_bits[::-1][:-(m-1)] if m > 1 else decoded_bits[::-1]
    decoded_bits = BinaryUtils.list_to_binary_string(decoded_bits)
    return decoded_bits

import matplotlib.pyplot as plt

def visualize_path_metrics(path_metrics_history, path_history):
    """Plot Viterbi state transitions and path metrics.

    Args:
        path_metrics_history (list[dict]): Metrics recorded at each time step.
        path_history (list[dict]): Survivor-path history.

    Returns:
        matplotlib.figure.Figure: Figure visualizing decoder evolution.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for t, history in enumerate(path_history):
        for state, (prev_state, bit) in history.items():
            if prev_state is not None:
                linestyle = '-' if bit == 0 else '--'
                x_values = [t, t+1]
                y_values = [int(''.join(map(str, prev_state)), 2), int(''.join(map(str, state)), 2)]
                ax.plot(x_values, y_values, marker='o', linestyle=linestyle)
                ax.text(t+0.5, (y_values[0] + y_values[1])/2, f'{path_metrics_history[t][state]:.1f}', 
                        horizontalalignment='center', verticalalignment='center')
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(['00', '01', '10', '11'])
    ax.set_xticks(range(len(path_history)))
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('States')
    ax.set_title('Path History and Metrics Visualization')
    # plt.show()
    return fig


def flip_bit(encoded, N):
    """Flip random positions in a mutable encoded sequence.

    Args:
        encoded (list[int]): Mutable encoded bit list.
        N (int): Number of bit positions to flip.

    Returns:
        list[int]: Mutated encoded sequence.
    """
    flip_indices = random.sample(range(len(encoded)), N)
    for flip_index in flip_indices:
        encoded[flip_index] = 1 - encoded[flip_index]
    print(f"[Viterbi] Encoded sequence after flipping bits at {flip_indices}: {encoded}")
    return encoded

def hypothesis_extended_viterbi_decode(encoded_bits_with_preamble, generators, extend_decode_num=3, premble_num=3):
    """Generate candidate bit extensions from a partially decoded payload.

    Args:
        encoded_bits_with_preamble (str | list[int]): Encoded sequence that
            still includes preamble and check bits.
        generators (list[list[int]]): Generator matrix.
        extend_decode_num (int): Number of extra bits to enumerate at the tail.
        premble_num (int): Preamble order used to locate the payload.

    Returns:
        list[str]: Candidate decoded bit strings after extension.
    """
    if type(encoded_bits_with_preamble)== str:
        encoded_bits_with_preamble = BinaryUtils.binary_string_to_list(encoded_bits_with_preamble)

    premble_len = int(1 + ((premble_num + 1) * premble_num) // 2)
    premble_len  += len(Settings.CHECK_CODE)
    payload_bits = encoded_bits_with_preamble[premble_len:]
    
    m = len(generators[0])
    G = len(generators)
    
    state_pairs = list(itertools.product([0, 1], repeat=m-1))
    
    path_metrics = {s: float('inf') for s in state_pairs}
    path_metrics[tuple([0]*(m-1))] = 0
    
    path_history = [{} for _ in range(len(payload_bits)//G + 2)]
    path_history[0] = {s: (None, None) for s in state_pairs}
    
    wait_to_process = []
    time_step = 0
    
    for bit in payload_bits:
        wait_to_process.append(bit)
        if len(wait_to_process) == G:
            time_step += 1
            current_metrics = path_metrics.copy()
            new_metrics = {s: float('inf') for s in state_pairs}
            path_history[time_step] = {}
            
            for prev_state in state_pairs:
                if current_metrics[prev_state] == float('inf'):
                    continue
                for input_bit in [0, 1]:
                    next_state = (input_bit,) + prev_state[:-1]
                    outputs = []
                    for gen in generators:
                        xor = input_bit * gen[0]
                        for i in range(1, len(gen)):
                            xor ^= prev_state[i-1] * gen[i]
                        outputs.append(xor % 2)
                    metric = sum(r != o for r, o in zip(wait_to_process, outputs))
                    total_metric = current_metrics[prev_state] + metric
                    if total_metric < new_metrics.get(next_state, float('inf')):
                        new_metrics[next_state] = total_metric
                        path_history[time_step][next_state] = (prev_state, input_bit)
            path_metrics = new_metrics
            wait_to_process = []

    if len(wait_to_process) > 0:
        extend_decode_num+=1
    
    # candidates = sorted(path_metrics.items(), key=lambda x: x[1])[:2**extend_decode_num]
    candidates = sorted(path_metrics.items(), key=lambda x: x[1])[:1]
    final_candidates = []
    
    for state, _ in candidates:
        decoded_bits = []
        current_state = state
        valid = True
        for t in range(time_step, 0, -1):
            if current_state not in path_history[t]:
                valid = False
                break
            prev_state, input_bit = path_history[t][current_state]
            decoded_bits.append(input_bit)
            current_state = prev_state
        if not valid or current_state != tuple([0]*(m-1)):
            continue
        decoded_bits.reverse()
        
        possible_extensions = list(itertools.product([0, 1], repeat=extend_decode_num))
        for ext_bits in possible_extensions:
            ext_state = state
            
            state_enc = tuple([0]*(m-1))
            reencoded_payload = []
            for bit in decoded_bits:
                outputs = []
                for gen in generators:
                    xor = bit * gen[0]
                    for i in range(1, len(gen)):
                        xor ^= state_enc[i-1] * gen[i]
                    outputs.append(xor % 2)
                reencoded_payload.extend(outputs)
                state_enc = (bit,) + state_enc[:-1]
            
            encoded_ext = []
            for bit in ext_bits:
                outputs = []
                for gen in generators:
                    xor = bit * gen[0]
                    for i in range(1, len(gen)):
                        xor ^= ext_state[i-1] * gen[i]
                    outputs.append(xor % 2)
                encoded_ext.extend(outputs)
                ext_state = (bit,) + ext_state[:-1]
            
            _extended_bits = encoded_bits_with_preamble[:premble_len] + reencoded_payload + encoded_ext
            final_candidates.append(BinaryUtils.list_to_binary_string(_extended_bits))
            
        return final_candidates

if __name__ == "__main__":
    generators = [
        [1, 1, 1],  # 0b111 (7)
        [1, 0, 1],  # 0b101 (5)
        [1, 0, 0]   # 0b110 (6)
    ]
    constraint_length = 3
    
    input_data = [1,0,0,1,1,0,1,1]
    print(f"[Viterbi] Input data: {input_data}")
    
    encoded = convolutional_encode(input_data, generators, constraint_length)
    print(f"[Viterbi] Encoded result: {encoded}")
    encoded = flip_bit(encoded, 3)
    
    decoded = viterbi_decode(encoded, generators, constraint_length)
    print(f"[Viterbi] Decoded result: {decoded}")
