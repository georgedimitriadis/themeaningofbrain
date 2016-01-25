# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 19:41:53 2014

@author: IntelligentSystems
"""

# -*- coding: cp1252 -*-
from scipy import fft
from scipy import ifft
from scipy.signal import butter
from scipy.signal import buttord
from scipy.signal import lfilter
from scipy.signal import bessel

def butterfilter(ECGdata):
    """ Filters the data using IIR butterworth filter

        Description:
            Digital filter which returns the filtered signal using butterworth
            5th order low pass design. The cutoff frequency is 0-35Hz with 100Hz
            as sampling frequency.
        Input:
            ECGdata -- list of integers (ECG data)
        Output:
            lfilter(b,a,ECGdata)-- filtered data along one-dimension with IIR
                                      butterworth filter
    """        
    fs = 100.00
    f = 35.00
    N=5
    [b,a]=butter(N,f/fs)
    return lfilter(b,a,ECGdata)

def besselfilter(ECGdata):
    """ Filters the data using IIR bessel filter

        Description:
            Digital filter which returns the filtered signal using bessel
            4th order low pass design. The cutoff frequency is 0-35Hz with 100Hz
            as sampling frequency.
        Input:
            ECGdata -- list of integers (ECG data)
        Output:
            lfilter(b,a,ECGdata)-- filtered data along one-dimension with IIR
                                     bessel filter
    """        
    fs = 100.00
    f = 35.00
    N=4
    [b,a]=bessel(N,f/fs)
    return lfilter(b,a,ECGdata)

def lowpassfilter(ECGdata):
    """ Filters the data using lowpass filter

        Description:
            Digital filter which returns the filtered signal using 60Hz
            lowpass filter. Transforms the signal into frequency domain
            using the fft function of the Scipy Module. Then, suppresses
            the 60Hz signal by equating it to zero. Finally, transforms
            the signal back to time domain using the ifft function.
        Input:
            ECGdata -- list of integers (ECG data)
        Output:
            ifft(fftECG) -- inverse fast fourier transformed array of filtered ECG data
    """        
    fftECG = fft(ECGdata)
    for i in range(len(fftECG)):
        if 590<i<910: fftECG[i]=0
    return ifft(fftECG)

def bandpass(ECGdata):
    """ Filters the data using bandpass filter

        Description:
            Digital filter which returns the filtered signal using 0.2-40Hz
            bandpass filter. Transforms the signal into frequency domain
            using the fft function of the Scipy Module. Then, suppresses
            the 60Hz signal by equating it to zero. Finally, transforms
            the signal back to time domain using the ifft function.
        Input:
            ECGdata -- list of integers (ECG data)
        Output:
            ifft(fftECG) -- inverse fast fourier transformed array of filtered ECG data
    """        
    fftECG = fft(ECGdata)
    for i in range(len(fftECG)):
        if 375<i<1125 or i<0.75 or i>1499.25: fftECG[i]=0    
    return ifft(fftECG)

def notchfilter(ECGdata):
    """ Filters the data using notch filter

        Description:
            Digital filter which returns the filtered signal using 60Hz
            notch filter. Transforms the signal into frequency domain
            using the fft function of the Scipy Module. Then, suppresses
            the 60Hz signal by equating it to zero. Finally, transforms
            the signal back to time domain using the ifft function.
        Input:
            ECGdata -- list of integers (ECG data)
        Output:
            ifft(fftECG) -- inverse fast fourier transformed array of filtered ECG data
    """         
    fftECG = fft(ECGdata)
    for i in range(len(fftECG)):
        if 590<i<620 or 880<i<910: fftECG[i]=0
    return ifft(fftECG)

def hanning(ECGdata):
    """ Filters the data using hanning filter method

        Description:
            Digital filter which solves for the computes for the weighted moving average of
            a given array of data.The following is the equation used to implement the
            Hanning filter: 
            
		y(nT)=1/4(x(nT)+2x(nT-T)+x(nT-2T))
        Input:
            ECGdata -- list of prefiltered integers (ECG data)
        Output:
            smoothed -- smoothed ECG signal using hanning method
    """         
    smoothed=[]
    for i in range(len(ECGdata)):
        smoothed.append((ECGdata[i]+2*ECGdata[i-1]+ECGdata[i-2])/4)
    return smoothed

def Notch50hzFs100(ECGdata):
    """ Filters the data using 50Hz Notch filter

        Description:
            Digital filter which computes for the 2 point moving average of
            the ECG data with 100 sampling frequency. Python translated code
            of the 50Hz Notch Filter C code of the EMI12 OEM development kit.
        Input:
            ECGdata -- list of integers (ECG data)
        Output:
            filteredECG -- filtered ECG data
    """        
    filteredECG=[]
    NOTCH_LENGTH_AVERAGE_FS100 = 100/50
    for i in range(len(ECGdata)):
        filteredECG.append((sum(ECGdata[i:(i+2)])/NOTCH_LENGTH_AVERAGE_FS100))
    return filteredECG




