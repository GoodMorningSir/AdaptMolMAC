"""Legacy multi-channel simulator backend for AdaptMolMAC.

The classes in this module are used internally by the higher-level simulation
API to synthesize received waveforms and per-channel SINR values.
"""

from .SimBase import simParams,ChannelModel,TxChips,TxBits, noiseParam
import numpy as np

class SimMmoTx:
    """Legacy multi-molecule transmission simulator used by `MCModel`.

    Attributes:
        interval (int): Symbol interval for convolution.
        yRx (list): Simulated received waveforms.
        yTx (list): Per-transmitter transmitted waveforms after channel mixing.
        SINR (list): Per-transmitter SINR estimates.
        txOffset (numpy.ndarray | None): Transmit offsets for each channel.
    """

    def __init__(self):
        """Initialize simulator components and default timing values."""
        
        self.NoiseVar = noiseParam()
        self.ChannelParam=ChannelModel()
        self.ChipParam=TxChips()
        self.MoBit=TxBits()
        
        self.interval  = 20
         
        self.yRx = simParams.nMo * [None]
        self.yTx = simParams.nMo * [simParams.nTx * [None]]
        self.SINR = simParams.nMo * [simParams.nTx * [None]]
        
        self.txOffset = None
        
        
        
    def setSimParams(self,NoiseVar=None,ChannelParam=None,ChipParam=None,MoBit=None):
        """Attach custom simulation components and generate transmit signals.

        Args:
            NoiseVar (noiseParam | None): Optional noise configuration object.
            ChannelParam (ChannelModel | None): Optional channel model.
            ChipParam (TxChips | None): Optional chip container.
            MoBit (TxBits | None): Optional transmit-bit container.
        """
        
        if NoiseVar is not None:
            self.NoiseVar = NoiseVar
        if ChannelParam is not None:
            self.ChannelParam = ChannelParam
        if ChipParam is not None:
            self.ChipParam = ChipParam
        if MoBit is not None:
            self.MoBit = MoBit
        
        self.genbits()
        self.genpreamble()
        self.genxtx()
        
        # self.plen = self.ChipParam.nSeq * self.ChipParam.Lp
        if self.txOffset is None:
            self.txOffset = np.array(simParams.nMo * [simParams.nTx * [0]],dtype=int)

    def genbits(self):
        """Generate random bits for the current simulation object."""
        self.MoBit.randomGenBits()
    def genpreamble(self):
        """Generate per-channel preamble chips."""
        self.MoBit.GenPreamble(self.ChipParam)
    def genxtx(self):
        """Generate the chip-expanded transmit sequences."""
        self.MoBit.GenxTx(self.ChipParam)
    
    def convolve(self, x, y):
        """Convolve a symbol sequence with one channel impulse response.

        Args:
            x (numpy.ndarray): Input symbol sequence.
            y (numpy.ndarray): Channel impulse response.

        Returns:
            numpy.ndarray: Convolution result aligned to the configured
            interval.
        """
        rval = np.array([0]*len(x)*(self.interval)+[0]*len(y)*2)
        for i in range(len(x)):
            temp = np.concatenate((np.array([0] * self.interval* i), y *x[i], np.zeros(len(rval) - len(y) - self.interval * i)))
            rval = rval +temp
        return rval
    
    def simulation(self,AddNoise = True):
        """Run the full transmit, channel, and optional noise simulation.

        Args:
            AddNoise (bool): Whether to apply the configured noise model.
        """
        
        for j in range(simParams.nMo):
            yRxLen = 0
            
            for i in range(simParams.nTx):
                self.yTx[j][i] = self.convolve(self.MoBit.xTx[j][i], self.ChannelParam.y_cirs[j][i])
                self.yTx[j][i]=np.concatenate((np.zeros(self.txOffset[j][i]),self.yTx[j][i]))
                if(self.yTx[j][i].shape[0]>yRxLen):
                    yRxLen=self.yTx[j][i].shape[0]
                    
            self.yRx[j] = np.zeros(yRxLen)
            
            for i in range(simParams.nTx):
                self.yRx[j][:self.yTx[j][i].shape[0]] += self.yTx[j][i]
            self.yRx[j] = np.concatenate((self.yRx[j] , np.zeros(self.txOffset[j][i])))
            
            if AddNoise:
                self.yRx[j]=self.NoiseVar.AddNoise(self.yRx[j])
            
                pSig = np.linalg.norm(self.yRx[j])
                for i in range(simParams.nTx):
                    yIn = self.yRx[j].copy()
                    yIn[:len(self.yTx[j][i])] -= self.yTx[j][i]
                    pIN = np.linalg.norm(yIn)
                    self.SINR[j][i] = 10 * np.log10(pSig / pIN)
            else:
                for i in range(simParams.nTx):
                    self.SINR[j][i] = 0
                    
    def DrawPic(self, numMo=0):
        """Plot one simulated received signal.

        Args:
            numMo (int): Molecule index to visualize.
        """
        import matplotlib.pyplot as plt
        plt.figure()
        x_time = self.ChannelParam.GenrateXtime_fromYcir(self.yRx[numMo], self.ChannelParam.unit_time)
        plt.plot(x_time.tolist(), self.yRx[numMo].tolist())
        plt.title(f'yRx{numMo} Plot')
        plt.xlabel('time')
        plt.ylabel('Value')
        plt.grid(True)
        plt.show()
    
    def retrunDict(self):
        """Return the simulator state as a dictionary.

        Returns:
            dict: Snapshot of the simulator configuration and results.
        """
        return {
            'NoiseVar': self.NoiseVar,
            'ChannelParam': self.ChannelParam,
            'ChipParam': self.ChipParam,
            'MoBit': self.MoBit,
            
            'plen': self.plen,
            'txOffset': self.txOffset,
            
            'yRx': self.yRx,
            'yTx': self.yTx,
            'SINR': self.SINR
        }
