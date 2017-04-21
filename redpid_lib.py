"""
RED PID LIB

where are you saved?

Library for red pitaya temperature lock

written for python3
@author: Gerhard Schunk
April 2016
"""
import numpy as np
import sys, os
from numpy import NaN, Inf, arange, isscalar, array, asarray
import ntpath
    
def ntest(lda,T,pol=0,MgO=0,MgO_th=5.0,E = 0): return 1.5

def path_leaf(path):
    head, tail = ntpath.split(path)
    tail =  ('.').join(tail.split('.')[:-1])
    return tail or ntpath.basename(head)
    
def tiltcorrect(ydata,windowlen=100,degree = 1):
#    import poly_iter
    x_ground = np.arange(len(ydata))
    x_ground_fit = smooth(x_ground,window_len=int(len(x_ground)/windowlen),window='hanning')
    ydata_fit    = smooth(ydata   ,window_len=int(len(ydata)   /windowlen),window='hanning')
    coeff = np.polyfit(x_ground_fit,ydata_fit, degree)
    
#    if degree == 1:
#        ytilt = tuple(x*coeff[0] + coeff[1] for x in x_ground)
#    if degree == 2:
#        ytilt = tuple(x*x*coeff[0] + x*coeff[1] + coeff[2]  for x in x_ground)
#    if degree == 3:
#        ytilt = tuple(x*x*x*coeff[0] + x*x*coeff[1] + x*coeff[2] + coeff[3]  for x in x_ground)
    ytilt = np.array([], dtype='double')
    for xx in x_ground:
        ytilt = np.append(ytilt,poly_horner(np.flipud(coeff),xx))
    return  ydata - ytilt 

def poly_horner(A, x):
    p = A[-1]
    i = len(A) - 2
    while i >= 0:
        p = p * x + A[i]
        i -= 1
    return p
    
def poly_naive(x, coeff):
    result = coeff[0]
    for i in range(1, len(coeff)):
        result = result + coeff[i] * x**i
    return result
    
def poly_iter(A, x):
    p = 0
    xn = 1
    for a in A:
        p += xn * a
        xn *= x
    return p


def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    http://wiki.scipy.org/Cookbook/SignalSmooth
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    np.hanning, np.hamming, np.bartlett, np.blackman, np.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y[(window_len/2-1):-(window_len/2)]
    


def peakdet(v, delta, x = None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html
    https://gist.github.com/endolith/250860
    Returns two arrays
    
    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %      
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.
    
    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.
    
    """
    maxtab = []
    mintab = []
       
    if x is None:
        x = arange(len(v))
    
    v = asarray(v)
    
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
    
    if not isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    
    if delta <= 0:
        sys.exit('Input argument delta must be positive')
    
    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN
    
    lookformax = True
    
    for i in arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        
        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return array(maxtab), array(mintab)

def nu2lambda(x):
    #for example: lambda in nm to frequency in GHz
    cms = 299792458 # speed of light
    return cms/x
    
def lambda2nu(x):
    #for example: lambda in nm to frequency in GHz
    cms = 299792458 # speed of light
    return cms/x
  
def double_exponential(t=range(1,100),t0=0,t_d=1,amp=1,y0=1):
    import math
    return amp * math.exp(-abs((t - t0) / t_d)) + y0
				
def lorentz(x=range(1,100),x0=0,gamma=1,amp=1,y0=1):
    return amp / (1.0 + 4*((x - x0) / gamma)**2 ) + y0
    
def gauss(x, *p):
    A, mu, sigma, y0 = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2)) + y0
    
def lowpass(cutoff, fs, order=5):
    import numpy as np
    from scipy.signal import butter, lfilter, freqz
    import matplotlib.pyplot as plt
    
    def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a
    def butter_lowpass_filter(data, cutoff, fs, order=5):
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y

    # Filter requirements.
    order = 6
    fs = 30.0       # sample rate, Hz
    cutoff = 3.667  # desired cutoff frequency of the filter, Hz
    
    # Get the filter coefficients so we can check its frequency response.
    b, a = butter_lowpass(cutoff, fs, order)
    
    # Plot the frequency response.
    w, h = freqz(b, a, worN=8000)
    plt.subplot(2, 1, 1)
    plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
    plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
    plt.axvline(cutoff, color='k')
    plt.xlim(0, 0.5*fs)
    plt.title("Lowpass Filter Frequency Response")
    plt.xlabel('Frequency [Hz]')
    plt.grid()

def read_data(filename,delim = ' '):
# Funktion zum Einlesen der selbsterzeugten kalibrationsdatein
    import csv
    rawdata=[]
	
    reader = csv.reader(open(filename, 'r'), delimiter=delim)#, quotechar='|')

    for line in reader:
		#print len(line)	
          rawdata.append(line)
	#ardata = np.array(rawdata, dtype='float64')
    return rawdata#ardata
    
def read_agilent_DSOx2024a(expdatfolder,expfilenr=0,expfiletype='csv',grdsubtract=False):
#This function read a measured frequeny spectrum and prepares it a bit  
#
#@author: Gerhard Schunk  
    from Ghad_lib import read_data, smooth, path_leaf
    import numpy as np
    import csv
    import glob
    
    headerlines = 4
    x1 = np.array([], dtype='float64')
    y1 = np.array([], dtype='float64')
    y2 = np.array([], dtype='float64')
    y3 = np.array([], dtype='float64')
    y4 = np.array([], dtype='float64')
    
    if expfiletype=='csv':
        expfilepaths = glob.glob(expdatfolder + '//*.csv')
        expfilename =  path_leaf(expfilepaths[expfilenr])   
        reader = csv.reader(open(expfilepaths[expfilenr], 'r'), delimiter=',')#, quotechar='|')
        
        
        line_count = 0
        for line in reader:
            if headerlines > line_count:
                line_count = line_count + 1
                continue
            x1 = np.append(x1,float(line[0]))
            y1 = np.append(y1,float(line[1]))
            y2 = np.append(y2,float(line[2]))
            y3 = np.append(y3,float(line[3]))
            y4 = np.append(y4,float(line[4]))
            
    
    print('exp data name: ', expfilename, '\n')

    if grdsubtract:
        groundsmoothlen = int(round(len(yfreq)/10))
        yfreq_ground = smooth(yfreq,groundsmoothlen,window='hanning')
        yfreq_ground = yfreq_ground[0:len(yfreq)]
        yfreq_filter = yfreq - yfreq_ground
        for ite in range(yfreq_filter.size):
            if yfreq_filter[ite]<0:
                yfreq_filter[ite] = 0
                
    return  x1, y1, y2, y3, y4, expfilename              
#    return xfreq_GHz, yfreq, expfilename
                
def autocorr(x):
    "calculates the autocorrelation function for multiple purposes"
#    result = np.convolve(x, x, mode='same')
    result = np.convolve(x, np.flipud(x), mode='full')
    return result[result.size/2.:]
				
def crosscorr(x,y):
    "calculates the crosscorrelation function for multiple purposes"
#    result = np.convolve(x, x, mode='same')
    result = np.convolve(x, np.flipud(y), mode='full')
    return result[result.size/2.:]
				
#    return result
         
def get_error_correlation(trace_1,trace_2,setpoint_xcorr):
    "Takes TE and TM traces and gives an output "
    current_corr = autocorr(trace_1,trace_2)
    error_corr = autocorr(current_corr,setpoint_xcorr)
    ind_max = np.argmax(error_corr)
    return ind_max

def get_error_max(trace_1,ind_set):
    "Takes maximum of trace and calculates error max"
    ind_error_max = np.argmax(trace_1) - ind_set
    return ind_error_max
				
def read_user_input(newstdin,flag,pid_status,P_pid,I_pid,G_pid):
    
    newstdinfile = os.fdopen(os.dup(newstdin))
    
    while True:
        stind_checkout = newstdinfile.readline()
        stind_checkout = stind_checkout[:len(stind_checkout)-1]
								
        if (stind_checkout == "on"):
	        print('Start the lock')
	        print(stind_checkout)
	        pid_status.value = 1   

        if (stind_checkout == "off"):
	        print('Stop the lock')
	        print(stind_checkout)
	        pid_status.value = 0   
									
        if (stind_checkout[0] == "P"):
	        print('New P', stind_checkout[1:]) 
	        P_pid.value = int(stind_checkout[1:])    
 
        if (stind_checkout[0] == "I"):
	        print('New I', stind_checkout[1:]) 
	        I_pid.value = int(stind_checkout[1:])    
									
        if ('New G', stind_checkout[0] == "G"):
	        print(stind_checkout[1:]) 
	        G_pid.value = int(stind_checkout[1:])

        if (stind_checkout == "exit"):
	        print('Stop the music')
	        flag.value = 1
	        newstdinfile.close()
	        break
        else:
	        continue
