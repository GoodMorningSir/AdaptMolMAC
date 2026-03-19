"""Low-level simulation primitives for molecular communication signals.

This module contains shared simulation parameters, the analytical channel
model, noise injection logic, and helper containers used by the legacy
simulation backend.
"""

import numpy as np
from ..config import Settings

def ToPos(data):
    """Clamp negative values in an array-like signal to zero.

    Args:
        data (numpy.ndarray): Input signal.

    Returns:
        numpy.ndarray: Copy of the input signal with negative values clipped.
    """
    rval = data.copy()
    rval[rval < 0] = 0
    return rval

class simParams:
    """Global simulation dimensions and timing parameters.

    Attributes:
        nTx (int): Number of transmitters.
        nMo (int): Number of molecule receivers.
        T (int): Base time-scaling factor.
    """

    nTx = 1
    nMo = 1
    T = 100
    
    @staticmethod
    def set_params(nTx = None, nMo=None, T=None):
        """Update shared transmitter, receiver, and symbol timing settings.

        Args:
            nTx (int | None): Number of transmitters.
            nMo (int | None): Number of molecule receivers.
            T (int | None): Updated base time-scaling factor.
        """
        if nTx is not None:
            simParams.nTx = nTx
        if nMo is not None:
            simParams.nMo = nMo
        if T is not None:
            Settings.SIG_END  *= int(simParams.T / T)
            simParams.T = T
            
class noiseParam:
    """Shared noise parameters for the simulator.

    Attributes:
        noiseb (float): Constant background level added to each sample.
        noisen (float): Additive Gaussian noise scale.
        noisep (float): Signal-dependent noise scale.
    """
    
    noiseb=1.0
    noisen=0.1
    noisep=0.0
    
    @staticmethod
    def set_noise_params(noiseb, noisen, noisep):
        """Set the background, additive, and Poisson-like noise levels.

        Args:
            noiseb (float): Constant background level.
            noisen (float): Additive Gaussian noise scale.
            noisep (float): Signal-dependent noise scale.
        """
        noiseParam.noiseb = noiseb
        noiseParam.noisen = noisen
        noiseParam.noisep = noisep
        
    @staticmethod
    def AddNoise(raw_yRx):
        """Apply the configured noise model to a raw received signal.

        Args:
            raw_yRx (numpy.ndarray): Raw noiseless signal.

        Returns:
            numpy.ndarray: Noisy signal after all configured perturbations.
        """
        yRx = raw_yRx.copy()
        yRx += noiseParam.noiseb
        yRx += noiseParam.noisep * np.random.randn(len(yRx)) * np.sqrt(yRx)
        yRx = np.maximum(yRx, 0)
        yRx += np.random.randn(len(yRx)) * noiseParam.noisen
        return yRx
    
class ChannelModel:
    """Simulate the channel impulse response used by the package.

    Attributes:
        default_betas (list[float]): Default analytical channel parameters.
        Tmax (int): Maximum simulated time span.
        mode (str): Downsampling mode for the generated CIR.
    """
    
    default_betas=[3, 0.025, 1.5, 1, 0, 0.1]
    Tmax = 10
    mode = 'max'
    
    @staticmethod
    def betas_check(betas):
        """Validate the six-parameter channel coefficient vector.

        Args:
            betas (list[float]): Candidate coefficient vector.

        Raises:
            ValueError: If the coefficients are not numeric.
            ValueError: If the vector does not contain exactly six entries.
        """
        if not all(isinstance(x, (int, float)) for x in betas):
            raise ValueError("All elements in betas must be numbers")
        if not (isinstance(betas, list) and len(betas) == 6):
            raise ValueError("betas must be a list containing exactly 6 numbers")

    @staticmethod
    def set_default_betas(betas):
        """Replace the default channel coefficients used by new models.

        Args:
            betas (list[float]): Replacement coefficient vector.
        """
        ChannelModel.betas_check(betas)
        ChannelModel.default_betas = betas
        
    @staticmethod
    def channelmodel(b, t):
        """Evaluate the analytical channel response at times `t`.

        Args:
            b (list[float]): Six-parameter analytical channel description.
            t (numpy.ndarray): Sample times.

        Returns:
            numpy.ndarray: Channel impulse-response values at the requested
            times.
        """
        b0, b1, b2, b3, b4, b5 = b
        cir = np.zeros_like(t)
        valid_time = t > b5
        t_valid = t[valid_time] - b5

        cir[valid_time] = b0 * (4 * np.pi * b1 * t_valid) ** -1.5 * \
            np.exp(-((b2 - b3 * t_valid) ** 2 + (b4 * t_valid) ** 2) / (4 * b1 * t_valid))

        return cir
    def __init__(self, betas=None, Tmax=None, mode=None):
        """Build channel impulse responses for all molecule/transmitter pairs.

        Args:
            betas (list | None): Optional nested channel-parameter structure.
            Tmax (int | None): Optional maximum simulated time.
            mode (str | None): CIR sampling mode.

        Raises:
            ValueError: If the provided beta structure does not match the
                current simulation dimensions.
            ValueError: If an unsupported mode is requested.
        """
        
        if betas is None:
            self.betas = simParams.nMo*[ simParams.nTx * [ChannelModel.default_betas]]
        else:
            if not (isinstance(betas, list) and len(betas) == simParams.nMo):
                raise ValueError(f"betas must be a list containing exactly {simParams.nMo} elements")
            for beta_list in betas:
                if not (isinstance(beta_list, list) and len(beta_list) == simParams.nTx):
                    raise ValueError(f"Each element in betas must be a list containing exactly {simParams.nTx} elements")
                for beta in beta_list:
                    ChannelModel.betas_check(beta)
            self.betas = betas
        
        if Tmax is not None:
            self.Tmax = Tmax
        if mode is not None:
            if mode not in ["rand", "max", "orgin"]:
                raise ValueError("wrong sim_mc_cir mode")
            self.mode = mode
            
        self.y_cirs=simParams.nMo*[ simParams.nTx * [None]]
        self.unit_time = simParams.T * 1e-3 / 10
        
        self.simulate()
            
    @staticmethod
    def GenerateXtime_fromTmax(Tmax, unit_time):
        """Generate a time axis from a maximum time and step size.

        Args:
            Tmax (float): Maximum simulated time.
            unit_time (float): Time step.

        Returns:
            numpy.ndarray: Generated time axis.
        """
        return np.arange(0, Tmax, unit_time)
    @staticmethod
    def GenrateXtime_fromYcir(ycir,unit_time):
        """Generate a time axis matching a sampled CIR array.

        Args:
            ycir (numpy.ndarray): Sampled channel impulse response.
            unit_time (float): Time step.

        Returns:
            numpy.ndarray: Generated time axis.
        """
        return np.arange(0, len(ycir)*unit_time ,unit_time)
            
    def simulate(self):
        """Generate and trim channel impulse responses for the current setup.

        Raises:
            ValueError: If no detectable signal support is found in a channel.
        """
        idx = {
                "rand": np.random.randint(0, 10),
                "max": np.argmax(self.y_cirs[0][0]) % 10,
                "orgin": 0
            }[self.mode]
        
        x_time = ChannelModel.GenerateXtime_fromTmax(self.Tmax, self.unit_time)
        
        for j in range(simParams.nMo):
            for i in range(simParams.nTx):
                self.y_cirs[j][i] =ChannelModel.channelmodel(self.betas[j][i], x_time)
                
                
        if self.mode in ["rand", "max"]:
            self.unit_time *= 10
            for j in range(simParams.nMo):
                for i in range(simParams.nTx):
                        self.y_cirs[j][i] = self.y_cirs[j][i][idx::10]

        istart = np.inf
        iend = -np.inf
        for j in range(simParams.nMo):
            for i in range(simParams.nTx):
                start_indices = np.where(self.y_cirs[j][i] > 1e-3)
                if start_indices[0].size > 0:
                    istart = min(istart, start_indices[0][0])
                    iend = max(start_indices[0][-1], iend)
                    self.y_cirs[j][i] = self.y_cirs[j][i][istart:iend+1]
                    self.y_cirs[j][i] = np.insert(self.y_cirs[j][i], 0, 0)
                    self.y_cirs[j][i] = np.append(self.y_cirs[j][i], 0)
                else:
                    raise ValueError("No signal detected")

    def drawPic(self, cirNumMo, cirNumTx):
        """Plot one channel impulse response.

        Args:
            cirNumMo (int): Molecule index.
            cirNumTx (int): Transmitter index.
        """
        import matplotlib.pyplot as plt
        plt.figure()
        x_time = ChannelModel.GenrateXtime_fromYcir(self.y_cirs[cirNumMo][cirNumTx], self.unit_time)
        plt.plot(x_time.tolist(), self.y_cirs[cirNumMo][cirNumTx].tolist())
        plt.title(f'CIR Mo:{cirNumMo}, Tx:{cirNumTx} Plot')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.grid(True)
        plt.show()

class TxChips:
    """Placeholder chip-sequence container used by the legacy simulator.

    Attributes:
        Lp (int): Preamble chip length.
        nDegree (int): Reserved code degree parameter.
        xChip (list): Per-channel chip placeholders.
    """

    Lp = 4
    def gen_xChip(self):
        """Generate chip placeholders for each molecule/transmitter pair.

        Returns:
            list: Placeholder chip structure sized to the current simulation.
        """
        xChip = simParams.nMo* [simParams.nTx * [None]]
        # xChipAll = GoldCodeList(0, self.nDegree, 'advanced')
        # ind = np.random.permutation(len(list(combinations(range(xChipAll.nTx), simParams.nMo))) * math.factorial(simParams.nMo))[:simParams.nTx]
        # for i in range(simParams.nTx):
        #     indtemp = ind[i]
        #     indrem = list(range(xChipAll.nTx))
        #     for j in range(simParams.nMo):
        #         indthis = indtemp % len(indrem)
        #         xChip[j][i] = xChipAll.Chip[indrem[indthis]]
        #         indtemp //= len(indrem)
        #         indrem.pop(indthis)
        return xChip
    
    def __init__(self, Lp = None, nDegree = 5):
        """Initialize the chip container.

        Args:
            Lp (int | None): Optional preamble length override.
            nDegree (int): Reserved code degree parameter.
        """
        if Lp is not None:
            TxChips.Lp = Lp
        self.nDegree = nDegree
        self.xChip=self.gen_xChip()
        # self.nSeq = self.xChip[0][0].shape[0]
        self.nSeq = None
        
class TxBits:
    """Transmit-bit and chip-expanded waveform container.

    Attributes:
        nBits (int): Number of data bits per transmitter.
        xBits (list): Raw bipolar transmit bits.
        pChip (list): Generated preamble chips.
        xTx (list): Final nonnegative transmitted sequences.
    """

    def __init__(self, nBits=None):
        """Allocate transmit-bit storage for the current simulation size.

        Args:
            nBits (int | None): Number of bits per transmitter.
        """
        if nBits is None:
            self.nBits = 100
        else:
            self.nBits = nBits
        self.xBits = simParams.nMo* [simParams.nTx * [None]]
        self.pChip = simParams.nMo* [simParams.nTx * [None]]
        self.xTx = simParams.nMo* [simParams.nTx * [None]]
        
    def randomGenBits(self):
        """Randomly generate bipolar transmit bits."""
        for j in range(simParams.nMo):
            for i in range(simParams.nTx):
                self.xBits[j][i] = 2 * np.random.randint(0, 2, self.nBits) - 1
    @staticmethod
    def GeneratePreambleChip(xchipi, Lp):
        """Generate a preamble chip pattern from one chip sequence.

        Args:
            xchipi (numpy.ndarray): Chip sequence.
            Lp (int): Preamble length parameter.

        Returns:
            numpy.ndarray: Generated preamble chips.
        """
        ppm = np.concatenate((np.ones(Lp, dtype=int), -np.ones(Lp, dtype=int)))
        ppm = ppm.reshape(-1, 1)
        rval = (ppm @ xchipi[::2].reshape(1, -1)).T.flatten()
        return rval
    
    def GenPreamble(self ,txChips):
        """Build preamble chips for every transmitter.

        Args:
            txChips (TxChips): Chip container providing base chip sequences.
        """
        for j in range(simParams.nMo):
            for i in range(simParams.nTx):
                self.pChip[j][i] = TxBits.GeneratePreambleChip(txChips.xChip[j][i], TxChips.Lp)
                
    @staticmethod
    def GenerateDataChips(xbit, xchip):
            """Expand bipolar bits into a repeated chip-domain sequence.

            Args:
                xbit (array-like): Bipolar bit sequence.
                xchip (array-like): Chip sequence.

            Returns:
                numpy.ndarray: Expanded chip-domain waveform.
            """
            xbit = np.array(xbit).reshape(-1, 1)
            xchip = np.array(xchip).reshape(1, -1)
            rval = (xbit @ xchip).flatten()
            return rval
        
    def GenxTx(self, txChips):
        """Generate nonnegative transmitted waveforms for every channel.

        Args:
            txChips (TxChips): Chip container providing base chip sequences.
        """
        for j in range(simParams.nMo):
            for i in range(simParams.nTx):
                self.xTx[j][i] = TxBits.GenerateDataChips(self.xBits[j][i], txChips.xChip[j][i])
                self.xTx[j][i] = np.concatenate((txChips.xChip[j][i], self.xTx[j][i]))
                self.xTx[j][i] = ToPos(self.xTx[j][i])
