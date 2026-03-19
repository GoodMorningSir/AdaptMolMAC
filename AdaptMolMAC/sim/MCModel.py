"""High-level signal simulation wrappers used by AdaptMolMAC.

This module exposes a simplified interface for generating synthetic molecular
communication waveforms from binary payloads.
"""

from .SimBase import simParams,ToPos,ChannelModel
from .SimMmoTx import SimMmoTx

import numpy as np
import matplotlib.pyplot as plt

simParams.set_params(1, 1, 100)


draw_cir = True
draw_ber = True
draw_interval = True
draw_interval_ber = True


class simtx(SimMmoTx):
    """Small single-channel transmitter wrapper around the legacy simulator."""
    
    def __init__(self):
        """Initialize a simplified transmit object for one bit stream."""
        super().__init__()
        self.nBits = 0
        self.Bits = None
        self.plen = 0
        self.pBits = None
        
    def genbits(self):
        """Load the user-provided bipolar bits into the simulator."""
        self.MoBit.nBits = self.nBits
        
        assert(self.Bits is not None)
        self.MoBit.xBits[0][0] = self.Bits
        
    def genpreamble(self):
        """Prepend the configured preamble bits when present."""
        if self.plen!=0:
            self.MoBit.xBits[0][0] = np.concatenate(self.pBits, self.MoBit.xBits[0][0])
        
    def genxtx(self):
        """Convert the transmit bit stream into a nonnegative waveform."""
        self.MoBit.xTx[0][0] = ToPos(self.MoBit.xBits[0][0])

class MCModel:
    """Generate synthetic molecular communication signals from bit strings.

    Attributes:
        simTx (simtx): Underlying single-stream simulator.
        amplitude (float): Output scaling factor applied after simulation.
    """

    def __init__(self, ChannelParam = None):
        """Create a simulator with optional custom channel coefficients.

        Args:
            ChannelParam (list[float] | None): Optional channel parameter set.
        """
        if ChannelParam is not None:
            ChannelModel.betas_check(ChannelParam)
            ChannelModel.set_default_betas(ChannelParam)
        self.simTx = simtx()
        self.amplitude = 1.0
    
    def setInterval(self, interval):
        """Set the symbol interval used during waveform synthesis."""
        self.interval = interval
        self.simTx.interval = interval
    def setConsisTxOffset(self, txOffsetValue):
        """Set a fixed transmit offset for all channels."""
        self.tx_offset = txOffsetValue
        self.simTx.txOffset = np.array(simParams.nMo * [simParams.nTx * [txOffsetValue]], dtype=int)
    def setAmplitude(self, amplitude):
        """Set a post-simulation amplitude scaling factor."""
        self.amplitude = amplitude

    def send(self, bits, AddNoise=True):
        """Synthesize the received waveform for a binary payload.

        Args:
            bits (str): Binary payload to transmit.
            AddNoise (bool): Whether to apply the configured noise model.

        Returns:
            numpy.ndarray: Simulated received waveform.

        Raises:
            ValueError: If `bits` contains characters other than `0` and `1`.
        """
        if not all(bit in '01' for bit in bits):
            raise ValueError("Input bits must be a string containing only '0' and '1'")
        self.simTx.nBits = len(bits)
        self.simTx.Bits = np.array([1 if bit == '1' else -1 for bit in bits])

        self.simTx.setSimParams()
        self.simTx.simulation(AddNoise=AddNoise)

        yRx = self.simTx.yRx[0]
        yRx = self.amplitude * yRx
        return yRx

if __name__ == '__main__':
    
    # mc = MCModel()
    # mc.setInterval(0)
    # mc.setConsisTxOffset(50)
    # bits = '1'
    # yRx=mc.send(bits)
    # np.savetxt('yRx1_signal1.csv', yRx, delimiter=',')
    
    # sig1 = Sig1Model()
    # best_b, best_range = sig1.fit_model(yRx)
    # print(best_b)
    # print(best_range)
    # sig1.set_params(best_b)
    # model = sig1.get_model()
    
    # t_valid = np.arange(best_range[0], best_range[1] + 1)
    # plt.plot(t_valid, model(t_valid), label='Fitted Model')
    # plt.legend()
    
    mc1 = MCModel()
    mc1.setInterval(10)
    mc1.setConsisTxOffset(50)
    bits1 = '110000011111101010101'
    yRx = mc1.send(bits1)
    np.savetxt('../../yRx2.csv', yRx, delimiter=',')
    
    # mc2 = MCModel()
    # mc2.setInterval(15)
    # mc2.setConsisTxOffset(40)
    # bits2 = '100101000111'
    # yRx2 = mc2.send(bits2)
    
    # min_len = min(len(yRx1), len(yRx2))
    # yRx1 = yRx1[:min_len]
    # yRx2 = yRx2[:min_len]
    # yRx = yRx1 + yRx2
    
    x_time = list(range(len(yRx)))

    plt.plot(x_time, yRx)
    plt.title(f'yRx Plot')
    plt.xlabel('time')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()
    
