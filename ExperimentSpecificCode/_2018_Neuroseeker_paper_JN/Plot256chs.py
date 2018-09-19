import IO.ephys as ephys
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal

#File names-------------------------------------------------------------------------------------------------------------
#256ch probe recording 2017-02-08 thalmus
raw_data_file_ivm = r"Z:\labs2\kampff\j\Joana Neto\Neuroseeker256ch\Type1_Probe6\Jesse\2017-02-08\Datakilosort\nfilt512\amplifier2017-02-08T21_38_55\amplifier2017-02-08T21_38_55_int16.bin"
#256ch probe recording 2017-02-16 cerebelum
raw_data_file_ivm = r"Z:\labs2\kampff\j\Joana Neto\Neuroseeker256ch\Type1_Probe6\2017-02-16\Datakilosort\amplifier2017-02-16T15_37_59\amplifier2017-02-16T15_37_59_int16.bin"
#256ch probe recording 2017-02-22 striatum and
raw_data_file_ivm = r"Z:\labs2\kampff\j\Joana Neto\Neuroseeker256ch\Type1_Probe6\2017-02-22\Datakilosort\amplifier2017-02-23T17_29_48\amplifier2017-02-23T17_29_48_int16.bin"
#insertion perpedincular to cortex
raw_data_file_ivm = r"Z:\labs2\kampff\j\Joana Neto\Neuroseeker256ch\Type1_Probe6\2017-02-22\Datakilosort\amplifier2017-02-23T19_36_39\amplifier2017-02-23T19_36_39_int16.bin"


#Open data--------------------------------------------------------------------------------------------------------------
num_ivm_channels = 256
amp_dtype = np.int16
sampling_freq = 20000
filtered_data_type = np.float64
voltage_step_size = 0.195e-6
scale_uV = 1000000
samples = 500000
high_pass_freq = 500
raw_data_ivm = ephys.load_raw_data(raw_data_file_ivm, numchannels=num_ivm_channels, dtype=amp_dtype)
temp_unfiltered = raw_data_ivm.dataMatrix [:, 0:samples]
temp_unfiltered = temp_unfiltered.astype(filtered_data_type)


#High pass filter-------------------------------------------------------------------------------------------------------
def highpass(data,BUTTER_ORDER=3, F_HIGH=14250,sampleFreq=30000.0,passFreq=100.0):
    b, a = signal.butter(BUTTER_ORDER,(passFreq/(sampleFreq/2), F_HIGH/(sampleFreq/2)),'pass')
    return signal.filtfilt(b,a,data)
temp_filtered = highpass(temp_unfiltered, F_HIGH = (sampling_freq/2)*0.95, sampleFreq = sampling_freq, passFreq = high_pass_freq)
temp_filtered_uV = temp_filtered * scale_uV * voltage_step_size


# 5 seconds
time_samples = 100000.0
index1 = np.int(temp_filtered_uV.shape[1]/10*5) #index1 for T2
index2 = np.int(index1 + time_samples)  #index2 for T2
#index1 = 100000   #index1 for CR1, St2, CoP3
#index2 = 200000   #index2 for CR1, St2, CoP3


# Electrodes number
linha_17 = np.array([163,232,236,240,249,199,193,220,36,61,1,8,12,47,41])
linha_16 = np.array([162,212,206,244,202,198,229,222,29,25,2,5,16,46,42])
linha_15 = np.array([217,211,207,245,251,255,228,225,35,60,55,9,17,20,95])
linha_14 = np.array([161,233,237,241,203,197,192,223,37,58,54,6,51,45,40])
linha_13 = np.array([216,234,205,242,201,195,218,224,28,24,3,13,50,21,39])
linha_12 = np.array([215,210,204,246,252,231,227,32,63,59,53,14,18,22,94])
linha_11 = np.array([160,235,238,247,200,196,219,33,27,57,7,10,48,44,38])
linha_10 = np.array([213,209,239,248,253,194,221,30,26,0,4,11,49,43,93])
linha_9 = np.array([168,159,154,181,145,140,191,131,125,120,67,111,106,77,97])
linha_8 = np.array([172,175,153,182,185,139,167,164,90,119,114,71,105,100,81])
linha_7 = np.array([169,157,179,148,143,189,134,129,123,65,68,109,75,78,84])
linha_6 = np.array([173,156,151,183,142,137,166,128,122,117,69,108,103,79,85])
linha_5 = np.array([170,176,152,147,186,138,133,127,89,118,113,72,104,99,82])
linha_4 = np.array([171,177,180,146,187,190,132,126,121,66,112,107,76,98,86])
linha_3 = np.array([174,155,150,184,141,136,165,91,88,116,70,73,102,80,83])
linha_2 = np.array([158,178,149,144,188,135,130,124,64,115,110,74,101,96,87])
linha_1 = np.array([214,208,243,250,254,230,226,34,62,56,52,15,19,23,92])
concatenate = np.concatenate((linha_17,linha_16,linha_15,linha_14,linha_13,linha_12,linha_11,linha_10,linha_9,linha_8,linha_7,linha_6,linha_5,linha_4,linha_3,linha_2,linha_1),axis=0)


#Plot 200 ms traces in 2D position
#Figure 4 Supplementary material
plt.figure()
for i in np.arange(np.shape(concatenate)[0]):
    plt.subplot(17, 15, i+1)
    plt.plot(temp_filtered_uV[concatenate[i], index1:index2].T , color = 'k')
    plt.ylim(-100, 100)
    # striatum
    #plt.xlim(18000,22000)
    # perpendicular
    #plt.xlim(5000,9000)
    # cerebellum
    #plt.xlim(12000,16000)
    # thalamus
    plt.xlim(90000,94000)
    plt.axis("OFF")








































#code not in use

#Plot
plt.figure()
for i in np.arange(np.shape(concatenate)[0]):
    plt.subplot(17, 15, i+1)
    plt.plot(temp_filtered_uV[concatenate[i], index1:index2].T , color = 'k')
    plt.ylim(-100, 100)
    # cortexlayer5
    #plt.xlim(25000,29000)
    # plt.xlim(65000,72000)
    # plt.xlim(70000,72000)
    # striatum
    #plt.xlim(32000,36000)
    #plt.xlim(18000,22000)
    #plt.xlim(53000,57000)
    # perpendicular
    plt.xlim(5000,9000)
    # cerebellum
    #plt.xlim(12000,16000)
    ##plt.xlim(12000,20000)
    # cerebellum2
    # plt.xlim(46000,50000)
    # cerebellum3
    # plt.xlim(26000, 30000)
    # cortex layer23
    #plt.xlim(76000,80000)
    # hipocampus1
    # plt.xlim(70000,74000)
    # hippocampus2
    # plt.xlim(40000,44000)
    # hipo3
    # plt.xlim(36000,40000)
    # plt.xlim(40000,50000)
    # thalamus
    #plt.xlim(90000,94000)
    ##plt.xlim(93000, 94000)
    plt.axis("OFF")

#Plot traces previous code
time_samples = 100000.0
#index1 = np.int(temp_filtered_uV.shape[1]/10*5)
#index2 = np.int(index1 + time_samples)
index1=250000
index2=350000

linha_17 = np.array([163,232,236,240,249,199,193,220,36,61,1,8,12,47,41])
linha_16 = np.array([162,212,206,244,202,198,229,222,29,25,2,5,16,46,42])
linha_15 = np.array([217,211,207,245,251,255,228,225,35,60,55,9,17,20,95])
linha_14 = np.array([161,233,237,241,203,197,192,223,37,58,54,6,51,45,40])
linha_13 = np.array([216,234,205,242,201,195,218,224,28,24,3,13,50,21,39])
linha_12 = np.array([215,210,204,246,252,231,227,32,63,59,53,14,18,22,94])
linha_11 = np.array([160,235,238,247,200,196,219,33,27,57,7,10,48,44,38])
linha_10 = np.array([213,209,239,248,253,194,221,30,26,0,4,11,49,43,93])
linha_9 = np.array([168,159,154,181,145,140,191,131,125,120,67,111,106,77,97])
linha_8 = np.array([172,175,153,182,185,139,167,164,90,119,114,71,105,100,81])
linha_7 = np.array([169,157,179,148,143,189,134,129,123,65,68,109,75,78,84])
linha_6 = np.array([173,156,151,183,142,137,166,128,122,117,69,108,103,79,85])
linha_5 = np.array([170,176,152,147,186,138,133,127,89,118,113,72,104,99,82])
linha_4 = np.array([171,177,180,146,187,190,132,126,121,66,112,107,76,98,86])
linha_3 = np.array([174,155,150,184,141,136,165,91,88,116,70,73,102,80,83])
linha_2 = np.array([158,178,149,144,188,135,130,124,64,115,110,74,101,96,87])
linha_1 = np.array([214,208,243,250,254,230,226,34,62,56,52,15,19,23,92])

teste = np.concatenate((linha_1,linha_2,linha_3,linha_4,linha_5,linha_6,linha_7,linha_8,linha_9,linha_10,linha_11,linha_12,linha_13,linha_14,linha_15,linha_16,linha_17),axis=0)
teste =np.reshape(teste, (17,int(np.shape(teste)[0]/17)))


offset_microvolt = 200

plt.figure()
for linha in np.arange(np.shape(teste)[0]):
    linha_selected = teste[linha,:]
    plt.subplot(1, 17, linha+1)
    #plt.figure()
    for x in linha_selected:
        plt.plot(temp_filtered_uV[x,index1:index2].T + (offset_microvolt* (np.argwhere(linha_selected==x)[0,0])))
        plt.show()
        #plt.title(np.str((str(linha_selected), index1, index2)))
        plt.ylim(-200,300)
        #cortexlayer5
        #plt.xlim(25000,29000)
        #plt.xlim(65000,72000)
        #plt.xlim(70000,72000)
        #striatum
        #plt.xlim(32000,36000)
        #plt.xlim(18000,22000)
        #plt.xlim(53000,57000)
        #perpendicular
        plt.xlim(0,4000)
        #plt.xlim(5000,9000)
        #cerebellum
        #plt.xlim(12000,16000)
        #plt.xlim(12000,20000)
        #cerebellum2
        #plt.xlim(46000,50000)
        #cerebellum3
        #plt.xlim(26000, 30000)
        #cortex layer23
        #plt.xlim(76000,80000)
        #hipocampus1
        #plt.xlim(70000,74000)
        #hippocampus2
        #plt.xlim(40000,44000)
        #hipo3
        #plt.xlim(36000,40000)
        #plt.xlim(40000,50000)
        #thalamus
        #plt.xlim(90000,94000)
        #plt.xlim(93000,94000)
        plt.axis("OFF")
        #fig_filename = os.path.join(r"Z:\j\Joana Neto\Neuroseeker256ch\Type1_Probe6\Plots" + '\\' + str(linha_selected) + 'linha.png')
    #plt.savefig(fig_filename)
    #plt.close()



