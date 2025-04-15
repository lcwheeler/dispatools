from dispatools import load_pdata
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt

def get_ppm_scale(dic):    
    """Function to pull parameters from loaded processed data and calculate ppm scale.
    
    Parameters

    ----------
    
    dic: dict
        dictionary of data paramets from nmrglue.bruker.read_pdata
        
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
    """

    magnitude = np.sqrt(data[0]**2 + data[1]**2)
    
    return magnitude
    
    
def optimize_rotation_rms(data_path1, data_path2, ppm_region, step_deg=1, plot=True, plotname="RMS-Angles", origin=(0,0), angle_range=(-180,180), figsize=(8,4)):
    """Optimize phase rotation of dataset 2 to match dataset 1 using RMS error. 
    Returns the angle with lowest RMS, and optionally plots RMS vs angle. 
    
    Parameters

    ----------
    
    data_path1: str
        Path to the first Bruker processed data directory
    data_path2: str
        Path to the second Bruker processed data directory
    ppm_region: tuple
        Upper and lower limits for the ppm region of interest
    step_deg: float
        Step size for sweeping angles (can be non-integer)
    plot: bool
        Whether to generate the optional plots of optimized rotation
    plotname: str
    	Name for the plot file
    origin: tuple
        cartesian coordinates of origin for rotation plot
    angle_range: tuple
    	range of angles to sweep during RMSE minimization
    figsize: tuple
    	figure size for optional plot (inches)
    """

    # Load the datasets
    params1, spec1 = load_pdata(data_path1)
    ppm1  = get_ppm_scale(params1)
    
    params2, spec2 = load_pdata(data_path2)
    ppm2  = get_ppm_scale(params2)

    # scale the spec and project to 1D
    R1 = spec1[0]
    I1 = spec1[1]
    nc_proc1 = params1['procs']["NC_proc"]
    scale1 = 2**nc_proc1;
    SPEC1 = (R1 + 1*np.sqrt(-1+0j) * I1) * scale1

    R2 = spec2[0]
    I2 = spec2[1]
    nc_proc2 = params2['procs']["NC_proc"]
    scale2 = 2**nc_proc2;
    SPEC2 = (R2 + 1*np.sqrt(-1+0j) * I2) * scale2

    # Interpolate the spectra using a quadratic spline
    f_splineS1 = interp1d(ppm1, SPEC1, kind='quadratic')
    f_splineS2 = interp1d(ppm2, SPEC2, kind='quadratic')

    SPEC1 = f_splineS1(ppm1)
    SPEC2 = f_splineS2(ppm2)

    # Extract region
    lo = ppm_region[0];
    hi = ppm_region[1];
    idx1 = np.where((ppm1 >= lo) & (ppm1 <= hi))
    idx2 = np.where((ppm2 >= lo) & (ppm2 <= hi))

    spec1r = SPEC1[idx1]
    spec2r = SPEC2[idx2]

    # Run sanity check 
    N = min(len(spec1r), len(spec2r))
    spec1r = spec1r[0:N]
    spec2r = spec2r[0:N]

    # Sweep angles
    angles_deg = np.array(np.arange(angle_range[0], angle_range[1]+1, step_deg))
    rms_values = np.zeros(len(angles_deg))

    for k in range(0, len(rms_values)):
        theta = np.deg2rad(angles_deg[k])
        rotated = spec2r * np.exp(1*np.sqrt(-1+0j) * theta)
        rms_values[k] = np.sqrt(np.mean(np.abs(spec1r - rotated)**2))

    # Find best angle
    min_rms = np.min(rms_values)
    min_idx = np.where(rms_values == min_rms)
    best_angle = angles_deg[min_idx][0]

    # Plot the result
    if plot==True:

        # set up plot and gridspec
        fig = plt.figure(constrained_layout=True, figsize=(figsize[0],figsize[1]))
        gs = fig.add_gridspec(nrows=1, ncols=2)

        # generate the RMSE subplot
        f_ax1 = fig.add_subplot(gs[0,0])
        f_ax1.set_title('Optimal Angle (RMSE)')
        f_ax1.plot(angles_deg, rms_values, color="black");
        f_ax1.set_xlabel("Rotation Angle (degrees)")
        f_ax1.set_ylabel("RMS Error")
        #f_ax1.set_xlim(-180,180)
        plt.axvline(best_angle, lw=0.5, linestyle="--", color="blue")
        ytick_range = list(f_ax1.get_yticks())
        #print(ytick_range)
        #plt.text(x=best_angle+np.abs(0.05*best_angle), y=0.5*np.max(ytick_range), s=r"{}$^\circ$".format(best_angle), color="blue")
        plt.text(x=best_angle+np.abs(0.05*best_angle), y=np.median(ytick_range), s=r"{}$^\circ$".format(best_angle), color="blue")
        
        # rotate data for vizualation - this should plot over relevant ppm range
        data_rotated, dr_rotated, di_rotated = rotate(spec2, complex(origin[0], origin[1]), best_angle)

        # generate the rotated polar subplot
        f_ax2 = fig.add_subplot(gs[0,1])
        f_ax2.plot(dr_rotated[idx2], di_rotated[idx2], color="darkviolet")
        f_ax2.plot(R1[idx1], I1[idx1], color="orange")
        f_ax2.plot(R2[idx2], I2[idx2], color="teal")       
        f_ax2.set_xlabel("Real (a.u.)")
        f_ax2.set_ylabel("Imaginary (a.u.)")
        f_ax2.set_title('Optimal Rotation')
        plt.axhline(0, lw=1, linestyle="--", color="gray")
        plt.axvline(0, lw=1, linestyle="--", color="gray")
        plt.gca().set_aspect("equal")
        plt.legend({"rotated":"darkviolet", "reference":"orange", "unrotated":"teal"})
        
        plt.savefig(plotname+".png", dpi=300)
        plt.savefig(plotname+".pdf")

        
    return best_angle, min_rms, angles_deg, rms_values, data_rotated
