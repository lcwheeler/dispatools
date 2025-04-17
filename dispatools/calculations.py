from dispatools import load_pdata
from scipy.interpolate import interp1d
from scipy.signal import argrelmin
import numpy as np
import matplotlib.pyplot as plt


def get_ppm_scale(dic):    
    """Function to pull parameters from loaded processed data and calculate ppm scale.
    
    Parameters

    ----------
    
    dic: dict
        dictionary of data parameters from nmrglue.bruker.read_pdata
        
    Returns
    -------
    ppm : numpy.array
        ppm scale matched to the real 1D input spectrum
        
    """

    offset_ppm = dic["procs"]["OFFSET"]   # ppm
    sw_Hz      = dic["procs"]["SW_p"]     # Hz
    sf_MHz     = dic["procs"]["SF"]       # MHz
    si         = dic["procs"]["SI"]      # points

    dppm = sw_Hz / sf_MHz / (si - 1)         # ppm per point
    ppm = offset_ppm - dppm * np.array(range(0,si))       # decreasing ppm axis

    return ppm 
    
 
def rotate(data, origin, angle):
    """Function to rotate a set of points by a specified angle.
    
     Parameters

    ----------
    
    data: numpy.array like
        numpy array from NMRGlue read_pdata() containing real and imaginary components
    origin: tuple
        origin in cartesian coordinates
    angle: float 
        Angle in degrees
        
    Returns
    -------
    rpoints : numpy.array
        Rotated data points
    rpr : numpy.array
        Real component of rotated data points
    rpi : numpy.array
        Imaginary component of rotated data points
    """

    points = np.array([complex(data[0][i],data[1][i]) for i in range(len(data[0]))])
    theta = np.deg2rad(angle)
    
    rpoints = (points - origin) * np.exp(complex(0, theta)) + origin
    
    rpr = rpoints.real
    rpi = rpoints.imag
    return rpoints, rpr, rpi
    

def magnitude_transformation(data):
    """Convert the real and imaginary components of NMR data in Bruker format to Magnitude Mode.
    
    Parameters

    ----------
    
    data: numpy.array like
        numpy array from NMRGlue read_pdata() containing real and imaginary components
        
    Returns
    -------
    magnitude : numpy.array
        Spectrum converted to magnitude mode
    """

    magnitude = np.sqrt(data[0]**2 + data[1]**2)
    
    return magnitude


