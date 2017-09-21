"""
return the correlation coefficient given two traces
@Dezhi
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import csv
from scipy.signal import butter, lfilter
from scipy.stats import pearsonr as corr
from scipy.signal import resample

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def get_corr(ts1, ts2, period):
    '''
        ts1 and ts2 are the two traces for computing the correlation
        period is how long the trace is in unit of day
    '''
    # resample the time-series to have equal length
    if len(ts1)==len(ts2):
        pass
    elif len(ts1)>len(ts2):
        ts1 = resample(ts1,len(ts2))
    else:
        ts2 = resample(ts2,len(ts1))

    # parameter for filter
    fs = float(len(ts1))/(period*24*3600)
    lowcut = 1.0/(6*3600) #6hrs
    highcut = 1.0/(20*60) #20mins
    out_ts1 = butter_bandpass_filter(ts1, lowcut, highcut, fs, order=3)
    out_ts2 = butter_bandpass_filter(ts2, lowcut, highcut, fs, order=3)

    #return the corr-coef
    return corr(out_ts1, out_ts2)[0]

if __name__ == "__main__":
    '''
        test case

    '''
    path1 = "/Users/hdz_1989/Documents/Dropbox/SDB/123/415/co2.csv"
    path2 = "/Users/hdz_1989/Documents/Dropbox/SDB/123/415/temperature.csv"
    rd1 = [float(i.strip('\n').strip()) for i in open(path1,'r').readlines()]
    rd2 = [float(i.strip('\n').strip()) for i in open(path2,'r').readlines()]
    # adujst to data to an array of even length
    ts1 = np.asarray(rd1)
    ts2 = np.asarray(rd2)
    print get_corr(ts1, ts2, 1)

