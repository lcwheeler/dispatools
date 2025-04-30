from dispatools import load_pdata, get_ppm_scale, magnitude_transformation, rotate
from scipy.interpolate import interp1d
from scipy.signal import argrelmin
import numpy as np
import matplotlib.pyplot as plt

def fidgen(shift, width, n, dw=0.1): 
    """Function to generate FID from specified shift, width using quadrature detection - i.e. real and imag
    are equal to amplitude multiplied by cos/sin(2pi*shift*t).
    
    Parameters

    ----------

    shift: float
        phase shift of FID
    width: float
        FWHM for FID peak
    n: int
        number of points in FID
    dw: float
        dwell time (0.1 gives 10 Hz spectral width)

    Returns
    
    -------
    
    fid : numpy.array like
        The simulated fid
        
    """
    
    t = np.arange(0, (n-1)*dw, dw)
    
    fidamp = np.exp(-t*np.pi*width) # FID amplitude
    #exp decay of 1 Hz gives 1/pi Hz FWHM, so scale appropriately
    
    #now get real, imag components
    i = np.sqrt(-1+0j)
    fidr = fidamp/2*np.cos(2*np.pi*shift*t); # carrier at 0 Hz, so shift - 0 = shift
    fidi = fidamp/2*np.sin(2*np.pi*shift*t);
    fid = fidr + 1*i*fidi;

    return fid
    

def phaseshift(fid, dw, theta):
    """Function to apply corrected phase shift to FID.
    
    Parameters

    ----------

    fid: numpy.array like
        phase shift of FID
    dw: float
        dwell time (0.1 gives 10 Hz spectral width)
    theta: float
        angle (in degrees) to phase shift FID

    Returns
    
    -------
    
    fidps : numpy.array like
        The phase-shifted FID
        
    """
    
    # dw not used but prefer to 
    # specify as 2nd arg whenever fid used

    # phase shifts the FID by theta degrees
    i = np.sqrt(-1+0j)
    fidps = fid*np.exp(-1*i*theta*np.pi/180);

    return fidps


def specgen(fid, dw):
    """Function to generate complex spectrum from FID, assuming quadrature detection.
    
    Parameters

    ----------

    fid: numpy.array like
        phase shift of FID
    dw: float
        dwell time (0.1 gives 10 Hz spectral width)
    SNR: float
        desired signal-to-noise ratio

    Returns
    
    -------
    
    spec: numpy.array like
        The complex spectrum
    f: numpy.array like
        spectral frequency
        
    """
    
    # carrier implicitly assumed 0 Hz
    n = len(fid); # number of points in FID
    
    sp = (1/dw)/(n-1) # freq spacing in spectral points
    f = np.arange(-(1/dw)/2-sp/2, (1/dw)/2-sp/2, sp) # spectral frequency
    # this includes a point at 0 freq, and one extra point at neg freq
        
    specpre = np.fft.fft(fid)
    spec = np.zeros(n)
    
    # negative frequencies
    spec[0:int(n/2)] = specpre[int(n/2+1):n]
    # positive frequencies
    spec[int(n/2+1):n] = specpre[0:int(n/2)] 
    
    # no spectrum flipping - do this in plotting
    spec = np.flip(spec)
    f = np.flip(f)

    return spec, f
    
    
def addnoise(fid, dw, SNR):
    """Function to add Gaussian noise to simulated FID.
    
    Parameters

    ----------

    fid: numpy.array like
        input noiseless FID
    dw: float
        dwell time (0.1 gives 10 Hz spectral width)
    SNR: float
        desired signal-to-noise ratio

    Returns
    
    -------
    
    fid : numpy.array like
        The noise-added FID
        
    """
    
    n = len(fid);
    spec, f = specgen(fid, dw)
    
    # Use magnitude spectrum to be phase-independent
    # Multiply by sqrt(2) to be consistent with previous SNR definition
    maxi = np.max(np.abs(spec)) * np.sqrt(2)

    # RMS of Gaussian noise (randn) is 1
    # SNR = max intensity / rms(noise) / 2
    # Scale further by sqrt(n) as we are adding to complex FID
    s = maxi / SNR / 2 / np.sqrt(n)
    
    rnoise = np.random.randn(1, n) * s
    inoise = np.random.randn(1, n) * s

    # Check deviation of rnoise, inoise from ideal
    ract = np.sqrt(np.mean(rnoise**2)) * np.sqrt(n)
    iact = np.sqrt(np.mean(inoise**2)) * np.sqrt(n)

    # Scale noise accordingly
    sr = (maxi / SNR / 2) / ract
    si = (maxi / SNR / 2) / iact

    rnoise = rnoise * sr
    inoise = inoise * si

    i = np.sqrt(-1+0j)

    fid = fid + rnoise + 1*i*inoise

    return fid
    


    

