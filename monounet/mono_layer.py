import torch
import torch.nn as nn
import numpy as np


class Mono2D(nn.Module):
    # Adopted from:
    # Copyright (c) 1996-2009 Peter Kovesi
    # School of Computer Science & Software Engineering
    # The University of Western Australia
    # pk at csse uwa edu au
    # http://www.csse.uwa.edu.au/
    # 
    # Permission is hereby  granted, free of charge, to any  person obtaining a copy
    # of this software and associated  documentation files (the "Software"), to deal
    # in the Software without restriction, subject to the following conditions:
    # 
    # The above copyright notice and this permission notice shall be included in all
    # copies or substantial portions of the Software.
    # 
    # The software is provided "as is", without warranty of any kind.
    #
    #------------------------------------------------------------------------
    #
    # Modified by A. BELAID, 2013
    #
    # Description: This code implements a part of the paper:
    # Ahror BELAID and Djamal BOUKERROUI. "A new generalised alpha scale 
    # spaces quadrature filters." Pattern Recognition 47.10 (2014): 3209-3224.
    #
    # Ahror BELAID and Djamal BOUKERROUI. "Alpha scale space filters for 
    # phase based edge detection in ultrasound images." ISBI (2014): 1247-1250.
    #
    # Copyright (c), Heudiasyc laboratory, Compiègne, France.
    #
    #------------------------------------------------------------------------

    def __init__(
        self, in_channels: int, nscale: int = 1, sigmaonf: list = None, wls: list = None,
        return_phase: bool = True, return_phase_asym: bool = False, return_phase_sym: bool = False,
        return_ori: bool = False, return_input: bool = False, norm: str = "std",
        T: float = 0.0, cut_off: float = 0.5, g: int = 10, episilon: float = 0.0001,
        min_wl: float = 3.0, max_wl: float = 128.0,
        trainable: bool = True
        ):
        super(Mono2D, self).__init__()

        # Hyperparameters - these can be tuned
        self.in_channels = in_channels
        self.return_phase = return_phase
        self.return_phase_sym = return_phase_sym
        self.return_phase_asym = return_phase_asym
        self.return_input = return_input
        self.return_ori = return_ori
        self.trainable = trainable
        self.norm = norm

        assert nscale > 0
        self.nscale = nscale

        # Fixed parameters
        # According to Nyquist theorem, the smallest wavelength should be 2 pixels to avoid aliasing.
        # Pick 3 pixels to be totally certain.
        self.min_wl = min_wl
        # Heuristically, at least for knee cartilage, the max wavelength won't reach 128 pixels.
        # This is a temporary fix that will be investigated in future work
        self.max_wl = max_wl
        
        # Learned parameters
        self.wls = nn.Parameter(self.initialize_wls(wls), requires_grad=self.trainable)
        self.sigmaonf = nn.Parameter(self.initialize_sigmaonf(sigmaonf), requires_grad=self.trainable)
        
        # Parameters below are used to construct the largest possible low-pass filter
        # that quickly falls to zero at the boundaries. Cut-off frequency (normalized)
        # should be between 0 and 0.5 according to Nyquist theorem.
        # The larger the value of g, the sharper the transition to zero.
        self.cut_off = cut_off
        self.g = g
        self.T = nn.Parameter(torch.tensor(T), requires_grad=self.trainable)
        # Set a small value used throughout the layer to avoid division by zero
        self.episilon = episilon

    def forward(self, x):
        x = x.to(dtype=torch.float32)   # For stable FFT computation
        B, C, rows, cols = x.size()
        # Transform the input image to frequency domain
        IM = torch.fft.fft2(x).to(self.get_device())
        IM = IM.view(B, C, 1, rows, cols)   # Process each channel separately

        # Get filters
        H, lgf = self.get_filters(rows, cols)

        # Bandpassed image in the frequency domain
        IMF = IM * lgf

        # Bandpassed image in the spatial domain
        f = torch.fft.ifft2(IMF).real

        # Bandpassed monogenic filtering, real part of h contains convolution result with h1, 
        # imaginary part contains convolution result with h2
        h = torch.fft.ifft2(IMF * H)
        h1 = h.real
        h2 = h.imag
        h_Amp2 = h1 ** 2 + h2 ** 2          # Amplitude of the bandpassed monogenic signal
        An = torch.sqrt(f ** 2 + h_Amp2)    # Magnitude of Energy (Amplitude)

        # Compute the phase asymmetry (odd - even)
        symmetry_energy = torch.abs(f) - torch.sqrt(h_Amp2)
                
        # Compute the phase asymmetry and phase symmetry
        phase_sym = torch.sum(An * torch.clamp(symmetry_energy - self.T, min=0), dim=2) / (torch.sum(An, dim=2) + self.episilon)
        phase_asym = torch.sum(torch.clamp(-symmetry_energy - self.T, min=0), dim=2) / (torch.sum(An, dim=2) + self.episilon)
        
        # Sum all responses across all scales
        f = torch.sum(f, dim=2)
        h1 = torch.sum(h1, dim=2)
        h2 = torch.sum(h2, dim=2)
        h_Amp2 = torch.sum(h_Amp2, dim=2)

        # Orientation - this varies +/- pi
        ori = torch.atan2(-h2,h1)
        # ori = self.scale_max_min(ori) # Normalizing angles loses circularity

        # Feature type - a phase angle +/- pi.
        ft = torch.atan2(f, torch.sqrt(h1 ** 2 + h2 ** 2 + self.episilon))       # Add episilon for stable gradients
        
        out = []
        if self.return_input:
            out.append(x)
        if self.return_phase:
            out.append(ft)
        if self.return_ori:
            out.append(ori)
        if self.return_phase_sym:
            out.append(phase_sym)
        if self.return_phase_asym:
            out.append(phase_asym)

        out = torch.cat([o.reshape(B, -1, o.shape[-2], o.shape[-1]) for o in out], dim=1)
        
        if self.norm == "std":
            out = self.std_normalize(out)
        elif self.norm == "min_max":
            out = self.min_max_normalize(out)
        elif self.norm is None:
            pass
        else:
            raise ValueError(f"Invalid normalization method: {self.norm}")
        return out

    def get_filters(self, rows, cols):
        u1, u2, radius = self.mesh_range((rows, cols))
        # Get rid of the 0 radius value in the middle (at top left corner after
        # fftshifting) so that taking the log of the radius, or dividing by the
        # radius, will not cause trouble.
        radius[0,0] = 1.

        # Construct the monogenic filters in the frequency domain.  The two
        # filters would normally be constructed as follows
        #    H1 = i*u1./radius; 
        #    H2 = i*u2./radius;
        # However the two filters can be packed together as a complex valued
        # matrix, one in the real part and one in the imaginary part.
        # When the convolution is performed via the fft the real part of the result
        # will correspond to the convolution with H1 and the imaginary part with H2.
        H = (1j*u1 - u2) / radius

        # Construct a low-pass filter that is as large as possible, yet falls
        # away to zero at the boundaries.  All filters are multiplied by
        # this to ensure no extra frequencies at the 'corners' of the FFT are
        # incorporated as this can upset the normalisation process when
        # calculating phase symmetry
        lp = self.lowpassfilter([rows, cols], self.cut_off, self.g)
        
        # Compute the log-Gabor filter
        lgf = self.compute_logGabor(radius)
        # Apply low-pass filter
        lgf = lgf * lp
        # Set the value at the 0 frequency point of the filter back to zero (undo the radius fudge).
        lgf[..., 0, 0] = 0

        return H, lgf

            
    def compute_logGabor(self, radius):
        # Obtain the different scales wavelengths
        wls = self.get_wls()
        # Obtain the center frequencies
        fo = 1.0 / wls
        # Reshape fo to be broadcastable with radius
        fo = fo.view(self.in_channels, self.nscale, 1, 1)
        # The parameter sigmaonf is in the range -inf to inf. Rescale it to 0-1
        sigmaonf = torch.nn.functional.sigmoid(self.sigmaonf)
        # Construct the filter
        filter = torch.exp((-(torch.log(radius/fo)) ** 2) / (2 * torch.log(sigmaonf) ** 2))
        return filter.to(self.get_device())


    def lowpassfilter(self, sze, cutoff, n):
        # LOWPASSFILTER - Constructs a low-pass butterworth filter.

        # usage: f = lowpassfilter(sze, cutoff, n)

        # where: sze    is a two element vector specifying the size of filter 
        #             to construct [rows cols].
        #     cutoff is the cutoff frequency of the filter 0 - 0.5
        #     n      is the order of the filter, the higher n is the sharper
        #             the transition is. (n must be an integer >= 1).
        #             Note that n is doubled so that it is always an even integer.

        #                     1
        #     f =    --------------------
        #                             2n
        #             1.0 + (w/cutoff)

        # The frequency origin of the returned filter is at the corners.

        # See also: HIGHPASSFILTER, HIGHBOOSTFILTER, BANDPASSFILTER


        # Copyright (c) 1999 Peter Kovesi
        # School of Computer Science & Software Engineering
        # The University of Western Australia
        # http://www.csse.uwa.edu.au/

        # Permission is hereby granted, free of charge, to any person obtaining a copy
        # of this software and associated documentation files (the "Software"), to deal
        # in the Software without restriction, subject to the following conditions:

        # The above copyright notice and this permission notice shall be included in 
        # all copies or substantial portions of the Software.

        # The Software is provided "as is", without warranty of any kind.

        # October 1999
        # August  2005 - Fixed up frequency ranges for odd and even sized filters
        #             (previous code was a bit approximate)
        if cutoff < 0 or cutoff > 0.5:
            raise('cutoff frequency must be between 0 and 0.5')
            
        if n % 1 != 0 or n < 1:
            raise('n must be an integer >= 1')
        
        if len(sze) == 1:
            sze = (sze, sze)
        else:
            rows = sze[0]
            cols = sze[1]

        _, _, radius = self.mesh_range((rows, cols))
        # Compute the filter
        f = ( 1.0 / (1.0 + (radius / cutoff) ** (2*n)) )

        return f
    

    def mesh_range(self, size):
        rows, cols = size
        # Set up u1 and u2 matrices with ranges normalized to +/- 0.5
        # The following code adjusts things appropriately for odd and even values of rows and columns.
        if cols % 2:
            xrange = torch.arange(-(cols - 1) / 2, (cols) / 2) / (cols - 1)
        else:
            xrange = torch.arange(-cols / 2, cols/2) / cols
        
        if rows % 2:
            yrange = torch.arange(-(rows - 1) / 2, (rows) / 2) / (rows - 1)
        else:
            yrange = torch.arange(-rows / 2, rows/2) / rows

        # print("xrange: ", xrange.shape, "yrange: ", yrange.shape)
        
        # print("xrange: ", xrange)
        # # print("yrange: ", yrange)

        u1, u2 = torch.meshgrid(xrange, yrange, indexing='xy')

        # Quadrant shift to put 0 frequency at the corners
        u1 = torch.fft.ifftshift(u1).to(self.get_device())
        u2 = torch.fft.ifftshift(u2).to(self.get_device())

        # print("\n")
        # print("u1: ", u1)
        # print("u2: ", u2)

        # Matrix values contain frequency values as a radius from center (but quandrant shifted)
        radius = torch.sqrt(u1**2 + u2**2).to(self.get_device())

        return u1, u2, radius
    

    def initialize_sigmaonf(self, sigmaonf):
        if sigmaonf is None:
            # Choose a random value very close to zero (akin to choosing a sigmaonf very close to 0.5)
            return torch.randn(1) * 0.05
        else:
            assert sigmaonf > 0 and sigmaonf < 1
            # Transform sigmaonf to between -inf and inf by applying the inverse sigmoid function
            sigmaonf = torch.tensor(sigmaonf)
            return torch.log(sigmaonf / (1 - sigmaonf))
    
    def initialize_wls(self, wls):
        if wls is None:
            return torch.randn(self.in_channels, self.nscale)
        else:
            wls = np.asarray(wls)  # Convert to numpy array for comparison
            assert np.all(wls > 0)  # Cannot have a negative wavelength
            # Rescale the wavelengths to be between 0 and 1 for faster training
            wls = torch.tensor((wls - self.min_wl) / (self.max_wl - self.min_wl))
            return torch.log(wls / (1 - wls))

    def rescale_wls(self, wls):
        return self.min_wl + wls * (self.max_wl - self.min_wl)
    
    def get_wls(self):
        return self.rescale_wls(torch.nn.functional.sigmoid(self.wls))
    
    def get_sigmaonf(self):
        return torch.nn.functional.sigmoid(self.sigmaonf)

    def get_device(self):
        return self.parameters().__next__().device
    
    def min_max_normalize(self, x):
        x_min = torch.amin(x, dim=(-2, -1), keepdim=True)
        x_max = torch.amax(x, dim=(-2, -1), keepdim=True)
        return (x - x_min) / (x_max - x_min)
    
    def std_normalize(self, x):
        x_mean = x.mean(dim=(-2, -1), keepdim=True)
        x_std = x.std(dim=(-2, -1), keepdim=True)
        return (x - x_mean) / (x_std + self.episilon)
    
    def get_params(self):
        # return a dictionary of the parameters
        return {
            "nscale": self.nscale,
            "wls": self.get_wls().tolist(),
            "sigmaonf": self.get_sigmaonf().item(),
            "return_phase": self.return_phase,
            "return_ori": self.return_ori,
            "return_phase_sym": self.return_phase_sym,
            "return_phase_asym": self.return_phase_asym,
            "return_input": self.return_input,
            "cut_off": self.cut_off,
            "g": self.g,
            "T": self.T.item(),
            "min_wl": self.min_wl,
            "max_wl": self.max_wl,
            "episilon": self.episilon,
            "norm": self.norm,
            "trainable": self.wls.requires_grad,
        }
    
    def extra_repr(self) -> str:
        extras = [
            f"in_channels={self.in_channels}",
            f"out_channels={self.in_channels}",
            f"nscale={self.nscale}",
            f"norm={self.norm}",
        ]
        if self.return_input:
            extras.append("return_input=True")
        if self.return_phase:
            extras.append("return_phase=True")
        if self.return_ori:
            extras.append("return_ori=True")
        if self.return_phase_sym:
            extras.append("return_phase_sym=True")
        if self.return_phase_asym:
            extras.append("return_phase_asym=True")
        extras.append(f"trainable={self.trainable}")
        return ", ".join(extras)


class Mono2DV2(Mono2D):
    def __init__(self, in_channels: int = 1, **kwargs):
        self.in_channels = in_channels
        super().__init__(in_channels, **kwargs)

        self.out_channels = ((
            int(self.return_input)
            + int(self.return_phase)
            + int(self.return_ori)
            + int(self.return_phase_sym)
            + int(self.return_phase_asym)
        ) * self.nscale * self.in_channels)

    def forward(self, x):
        x = x.to(dtype=torch.float32)
        B, C, rows, cols = x.size()
        # Transform the input image to frequency domain
        IM = torch.fft.fft2(x, dim=(-2, -1), norm="ortho").to(self.get_device())    # FLOPS = 5 × H × W × log₂(H × W)

        # Get filters
        H, lgf = self.get_filters(rows, cols)

        # Bandpassed image in the frequency domain
        IMF = IM.view(B, C, 1, rows, cols) * lgf            # FLOPS = nscale × 6 × H × W => (a+ib) * (c+id) = (ac-bd) + i(ad+bc)

        # Bandpassed image in the spatial domain
        f = torch.fft.ifft2(IMF, dim=(-2, -1)).real           # FLOPS = 5 × H × W × log₂(H × W)

        # Bandpassed monogenic filtering, real part of h contains convolution result with h1, 
        # imaginary part contains convolution result with h2
        h = torch.fft.ifft2(IMF * H, dim=(-2, -1))            # FLOPS = [5 × H × W × log₂(H × W)] + [6 × H × W]
        h1 = h.real
        h2 = h.imag
        h_Amp2 = h1 ** 2 + h2 ** 2          # Amplitude of the bandpassed monogenic signal
        An = torch.sqrt(f ** 2 + h_Amp2)    # Magnitude of Energy (Amplitude)
        
        # Compute the phase asymmetry (odd - even)
        symmetry_energy = torch.abs(f) - torch.sqrt(h_Amp2)

        # Compute the phase asymmetry and phase symmetry
        phase_sym = (An * torch.clamp(symmetry_energy - self.T, min=0)) / (An + self.episilon)
        phase_asym = (torch.clamp(-symmetry_energy - self.T, min=0)) / (An + self.episilon)

        # # Orientation - this varies +/- pi
        ori = torch.atan2(-h2,h1)

        # Feature type - a phase angle +/- pi.
        ft = torch.atan2(f, torch.sqrt(h1 ** 2 + h2 ** 2 + self.episilon))       # Add episilon for stable gradients            # FLOPS = 1 × H × W
        
        out = []
        if self.return_input:
            out.append(x)
        if self.return_phase:
            out.append(ft)
        if self.return_ori:
            out.append(ori)
        if self.return_phase_sym:
            out.append(phase_sym)
        if self.return_phase_asym:
            out.append(phase_asym)

        # Concatenate along the channel dimension: (B x [features] x nscale x H x W) -> (B x (features*nscale) x H x W)
        out = torch.cat([o.reshape(B, -1, o.shape[-2], o.shape[-1]) for o in out], dim=1)
        # out = torch.stack(out, dim=1)
        if self.norm == "std":
            out = self.std_normalize(out)
        elif self.norm == "min_max":
            out = self.min_max_normalize(out)
        elif self.norm is None:
            pass
        else:
            raise ValueError(f"Invalid normalization method: {self.norm}")
        
        return out

    def compute_logGabor(self, radius):
        # Obtain the different scales wavelengths
        wls = self.get_wls()
        # Obtain the center frequencies
        fo = 1.0 / wls
        # Reshape fo to be broadcastable with radius
        fo = fo.view(self.in_channels, self.nscale, 1, 1)
        
        # The parameter sigmaonf is in the range -inf to inf. Rescale it to 0-1
        sigmaonf = torch.sigmoid(self.sigmaonf).view(self.in_channels, self.nscale, 1, 1)
        
        # Construct the filter
        ratio = radius / fo  # or add +1e-8 before log

        numerator = -(torch.log(ratio)) ** 2
        denominator = 2 * torch.log(sigmaonf) ** 2
        filter = torch.exp(numerator / denominator)
        # denom = (torch.log(sigmaonf) ** 2) + 1e-12
        # filter = torch.exp(-(torch.log(ratio) ** 2) / (2 * denom))

        # filter = torch.exp((-(torch.log(radius/fo)) ** 2) / (2 * torch.log(sigmaonf) ** 2))
        return filter.to(self.get_device())
    
    def initialize_sigmaonf(self, sigmaonf):
        if sigmaonf is None:
            # Choose random values very close to zero (akin to choosing sigmaonf ~ 0.5)
            return torch.randn(self.in_channels, self.nscale) * 0.05
        else:
            sigmaonf = torch.as_tensor(sigmaonf, dtype=torch.float32)
            if sigmaonf.ndim == 0:
                sigmaonf = sigmaonf.repeat(self.in_channels, self.nscale)
            else:
                assert sigmaonf.numel() == self.in_channels * self.nscale
            valid = torch.all((sigmaonf > 0) & (sigmaonf < 1))
            assert bool(valid)
            # Transform sigmaonf to between -inf and inf by applying the inverse sigmoid function
            return torch.log(sigmaonf / (1 - sigmaonf))
    
    def initialize_wls(self, wls):
        if wls is None:
            return torch.randn(self.in_channels, self.nscale)
        else:
            wls = np.asarray(wls)  # Convert to numpy array for comparison
            assert np.all(wls > 0)  # Cannot have a negative wavelength
            # Rescale the wavelengths to be between 0 and 1 for faster training
            wls = torch.tensor((wls - self.min_wl) / (self.max_wl - self.min_wl))
            return torch.log(wls / (1 - wls))
    
    def get_params(self):
        # return a dictionary of the parameters
        params = {
            "in_channels": self.in_channels,
            "nscale": self.nscale,
            "wls": self.get_wls().tolist(),
            "sigmaonf": self.get_sigmaonf().tolist(),
            "cut_off": self.cut_off,
            "g": self.g,
            "T": self.T.item(),
            "min_wl": self.min_wl,
            "max_wl": self.max_wl,
            "episilon": self.episilon,
            "norm": self.norm,
            "trainable": self.wls.requires_grad,
        }
        # Only include return_* keys if their values are True
        if self.return_phase:
            params["return_phase"] = True
        if self.return_ori:
            params["return_ori"] = True
        if self.return_phase_sym:
            params["return_phase_sym"] = True
        if self.return_phase_asym:
            params["return_phase_asym"] = True
        if self.return_input:
            params["return_input"] = True
        return params

    def extra_repr(self) -> str:
        return f"in_channels={self.in_channels}, out_channels={self.out_channels}, " + super().extra_repr()


class Mono2DV3(Mono2DV2):
    """
    Mono2DV3: learn n_freq base wavelengths per channel (center wl0), and generate the
    remaining nscale wavelengths geometrically using a learnable scaling factor.
    Bandwidth (sigmaonf) is learned per center frequency and shared across its scales.
    """
    def __init__(
        self,
        in_channels: int = 1,
        nscale: int = 3,
        n_freq: int = 1,
        factor: float = 2.5,
        **kwargs,
    ):
        self.in_channels = in_channels
        self.nscale = nscale
        self.n_freq = n_freq
        super().__init__(in_channels, nscale=nscale, **kwargs)
        self.factor = nn.Parameter(self.initialize_factor(factor), requires_grad=self.trainable)
        self.out_channels = ((
            int(self.return_input)
            + int(self.return_phase)
            + int(self.return_ori)
            + int(self.return_phase_sym)
            + int(self.return_phase_asym)
        ) * self.nscale * self.n_freq * self.in_channels)

    def initialize_sigmaonf(self, sigmaonf):
        # Learn one sigmaonf per center frequency (shared across its scales)
        if sigmaonf is None:
            return torch.randn(self.in_channels, self.n_freq) * 0.05
        else:
            sigmaonf = torch.as_tensor(sigmaonf, dtype=torch.float32)
            if sigmaonf.ndim == 0:
                sigmaonf = sigmaonf.repeat(self.in_channels, self.n_freq)
            else:
                sigmaonf = sigmaonf.reshape(-1)
                assert sigmaonf.numel() == self.in_channels * self.n_freq
                sigmaonf = sigmaonf.view(self.in_channels, self.n_freq)
            valid = torch.all((sigmaonf > 0) & (sigmaonf < 1))
            assert bool(valid)
            return torch.log(sigmaonf / (1 - sigmaonf))

    def initialize_wls(self, wls):
        # Learn n_freq base wavelengths per channel (wl0s)
        if wls is None:
            return torch.randn(self.in_channels, self.n_freq)
        else:
            wls = np.asarray(wls)
            assert np.all(wls > 0)
            wls = torch.as_tensor(wls, dtype=torch.float32).reshape(-1)
            if wls.numel() == 1:
                wls = wls.repeat(self.in_channels * self.n_freq)
            assert wls.numel() == self.in_channels * self.n_freq
            wls = wls.view(self.in_channels, self.n_freq)
            wls = (wls - self.min_wl) / (self.max_wl - self.min_wl)
            return torch.log(wls / (1 - wls))

    def initialize_factor(self, factor):
        """
        factor is the geometric ratio r constrained to (1, r_max):
        r = 1 + sigmoid(p) * (r_max - 1)
        """
        if self.nscale <= 1:
            return torch.zeros(1)

        r_max = float((self.max_wl / self.min_wl) ** (1.0 / (self.nscale - 1)))

        if factor is None:
            factor = r_max
        else:
            factor = float(factor)

        # keep inside (1, r_max)
        factor = max(1.0001, min(factor, r_max - 1e-6))

        alpha = (factor - 1.0) / (r_max - 1.0)  # in (0,1)
        alpha = np.clip(alpha, 1e-6, 1 - 1e-6)
        p = torch.log(torch.tensor(alpha, dtype=torch.float32) / (1 - torch.tensor(alpha, dtype=torch.float32)))
        return p.view(1)

    def get_factor(self):
        if self.nscale <= 1:
            return torch.tensor(1.0, device=self.factor.device, dtype=self.factor.dtype)
        alpha = torch.sigmoid(self.factor)  # (0,1)
        r_max = (self.max_wl / self.min_wl) ** (1.0 / (self.nscale - 1))
        r = 1.0 + alpha * (r_max - 1.0)
        return r


    def get_sigmaonf(self):
        # Broadcast per-center sigmaonf across scales
        s = torch.sigmoid(self.sigmaonf).clamp(1e-6, 1 - 1e-6)  # (C,n_freq)
        s = s.unsqueeze(-1).repeat(1, 1, self.nscale)          # (C,n_freq,nscale)
        return s.reshape(self.in_channels, self.n_freq * self.nscale)

    def get_wls(self):
        # geometric progression: wl_s = wl0 * r^s, with wl0 learned per center freq
        r = self.get_factor()                                       # scalar > 1
        wl0_max = self.max_wl / (r ** (self.nscale - 1))
        wl0_max = torch.clamp(wl0_max, min=self.min_wl + 1e-6)

        wl0 = torch.sigmoid(self.wls)                               # (C,n_freq) in (0,1)
        wl0 = self.min_wl + wl0 * (wl0_max - self.min_wl)           # (C,n_freq) in [min_wl, max_wl]

        exps = torch.arange(self.nscale, device=wl0.device, dtype=wl0.dtype).view(1, 1, self.nscale)
        wls = wl0.unsqueeze(-1) * (r ** exps)                       # (C,n_freq,nscale)
        return wls.reshape(self.in_channels, self.n_freq * self.nscale)

    def compute_logGabor(self, radius):
        # Same as Mono2DV2 but uses shared sigmaonf and geometric wls
        wls = self.get_wls()
        fo = 1.0 / wls
        fo = fo.view(self.in_channels, self.n_freq * self.nscale, 1, 1)

        sigmaonf = self.get_sigmaonf().view(self.in_channels, self.n_freq * self.nscale, 1, 1)

        ratio = radius / fo
        numerator = -(torch.log(ratio)) ** 2
        denominator = 2 * torch.log(sigmaonf) ** 2
        filter = torch.exp(numerator / denominator)
        return filter.to(self.get_device())

    def get_params(self):
        params = super().get_params()
        params["factor"] = self.get_factor().item()
        params["n_freq"] = self.n_freq
        return params

    def extra_repr(self) -> str:
        return f"n_freq={self.n_freq}, " + super().extra_repr()
