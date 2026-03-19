"""Minimal `main.py`-style example for AdaptMolMAC."""

import AdaptMolMAC as amm
amm.Logger.enable()


def decode_payload(encoded_bits, tx, stationary_processor, generator):
    """Strip preamble/check bits and recover the original payload."""
    preamble_len = len(stationary_processor.preamble) + len(amm.Settings.CHECK_CODE)
    encoded_payload = encoded_bits[preamble_len:len(tx.encode_data[0])]
    return amm.viterbi_decode(encoded_payload, generator, len(generator[0]))


def main():
    """Run the same pipeline as `main.py` with fixed two-transmitter inputs."""
    generator = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 0, 0],
    ]
    tx1_bits = "01010011010"
    tx2_bits = "10010011011"

    tx1 = amm.ChannelModel_Tx(
        interval=17,
        tx_offset=50,
        amplitude=1.0,
        n_preamble=amm.Settings.PREAMBLE_NUM,
        viterbi_gen=generator,
    )
    tx2 = amm.ChannelModel_Tx(
        interval=22,
        tx_offset=864,
        amplitude=1.0,
        n_preamble=amm.Settings.PREAMBLE_NUM,
        viterbi_gen=generator,
    )

    y_rx0 = amm.yRxData()
    y_rx0 += tx1.transmit(tx1_bits)
    y_rx0 += tx2.transmit(tx2_bits)

    signal_processor = amm.SignalProcessor()
    y_rx = signal_processor.process_signal(y_rx0, x_offset=1)
    filtered_y_rx = signal_processor.kalman_filter(y_rx)
    filtered_y_rx = signal_processor.adaptive_threshold_filter(filtered_y_rx)
    filtered_y_rx = signal_processor.retain_peak_points_filter(
        peak_points_num=3 + amm.Settings.PEAK_POINT_EXCE_CUT,
        yrx_data=filtered_y_rx,
    )

    stationary_processor = amm.StationaryProcessor(
        n_preamble=amm.Settings.PREAMBLE_NUM
    )
    channel_info, mse = stationary_processor.estimate_channel(
        y_rx,
        filtered_y_rx,
        interval_dev=0,
        start_pos_dev=1,
    )
    channel_info.reset_start_pos()

    decoder = amm.DynamicDecoder(channel_info.yRx, [channel_info], generator)
    decoded_channels = decoder.decode(MAX_SIGNAL_NUM=2)

    decoded_payloads = [
        decode_payload(bits, tx, stationary_processor, generator)
        for bits, tx in zip(decoded_channels, [tx1, tx2])
    ]

    print("Channel estimation MSE:", mse)
    print("Transmitted payloads:", [tx1_bits, tx2_bits])
    print("Decoded channel count:", len(decoded_channels))
    for i, bits in enumerate(decoded_channels):
        print(f"Decoded encoded channel {i}:", bits)
    print("Recovered payloads after Viterbi:", decoded_payloads)


if __name__ == "__main__":
    main()
