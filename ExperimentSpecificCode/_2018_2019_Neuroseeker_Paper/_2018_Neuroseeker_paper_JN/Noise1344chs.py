import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import scipy.stats as stats
import IO.ephys as ephys


#Calculation of the noise for 2 regions(G1 and G2), 212 electrodes------------------------------------------------------

#File names
#bin files are filtered with a band-pass of 500-3,500 Hz and
#when using the external reference the median signal within each group across the recording
#electrodes from the respective group was subtracted

#Internal reference data
#2regions ON
time = r'17_50_36.bin'
#4regions ON
time = r'17_52_36.bin'
#6regions ON
time =r'17_54_35.bin'
#8regions ON
time= r'17_56_25.bin'
#12regions ON
time =r'18_15_12.bin'
binary_data_filename = r'Z:\n\Neuroseeker Probe Recordings\Neuroseeker_2017_08_08\Data'
binary_data_filename= os.path.join(binary_data_filename, time)

#External reference data
#Calculation of the noise for 2 regions
#2regions ON
time = r'18_18_41.bin'
binary_data_filename=r"Z:\n\Neuroseeker Probe Recordings\Neuroseeker_2017_08_08\Data\18_18_41.bin"
#4regions ON
time = r'18_20_31.bin'
binary_data_filename=r"Z:\n\Neuroseeker Probe Recordings\Neuroseeker_2017_08_08\Data\18_20_31.bin"
#6regions ON
time =r'18_22_40.bin'
binary_data_filename = r"Z:\n\Neuroseeker Probe Recordings\Neuroseeker_2017_08_08\Data\18_22_40.bin"
#8regions ON
time= r'18_24_20.bin'
binary_data_filename =r"Z:\n\Neuroseeker Probe Recordings\Neuroseeker_2017_08_08\Data\18_24_20.bin"
#10regions ON
binary_data_filename = r"Z:\n\Neuroseeker Probe Recordings\Neuroseeker_2017_08_08\Data\18_26_30.bin"
time=r'18_26_30.bin'
#12regions ON
time =r'18_40_36.bin'
binary_data_filename = r'Z:\n\Neuroseeker Probe Recordings\Neuroseeker_2017_08_08\Data'
binary_data_filename= os.path.join(binary_data_filename, time)


#Open Data
number_of_channels_in_binary_file = 1440
sampling_frequency = 20000
raw_data = np.memmap(binary_data_filename, dtype=np.int16, mode='r')
number_of_timepoints_in_raw = int(raw_data.shape[0] / number_of_channels_in_binary_file)
raw_data = np.reshape(raw_data, (number_of_channels_in_binary_file, number_of_timepoints_in_raw), order='F')


#Electrode numbers, 212 AP traces
OFFchannels= np.arange(241,1441)
lfpchannels= np.array([10,30,50,70,90,110,130,150,170,190,210,230,250,270,290,310,330,350,370,390,410,430,450,470,490,510,530,550,570,590,610,630,650,670,690,710,730,750,770,790,810,830,850,870,890,910,930,950,970,990,1010,1030,1050,1070,1090,1110,1130,1150,1170,1190,1210,1230,1250,1270,1290,1310,1330,1350,1370,1390,1410,1430])
refchannels= np.array([57,58,59,60,61,62,63,64,177,178,179,180,181,182,183,184,297,298,299,300,301,302,303,304,417,418,419,420,421,422,423,424,537,538,539,540,541,542,543,544,657,658,659,660,661,662,663,664,777,778,779,780,781,782,783,784,897,898,899,900,901,902,903,904,1017,1018,1019,1020,1021,1022,1023,1024,1137,1138,1139,1140,1141,1142,1143,1144,1257,1258,1259,1260,1261,1262,1263,1264,1377,1378,1379,1380,1381,1382,1383,1384])
OUTchannels= np.array([1081,1082,1083,1084,1085,1086,1087,1088,1089,1090,1091,1092,1093,1094,1095,1096,1097,1098,1099,1100,1101,1102,1103,1104,1105,1106,1107,1108,1109,1110,1111,1112,1113,1114,1115,1116,1117,1118,1119,1120,1121,1122,1123,1124,1125,1126,1127,1128,1129,1130,1131,1132,1133,1134,1135,1136,1137,1138,1139,1140,1141,1142,1143,1144,1145,1146,1147,1148,1149,1150,1151,1152,1153,1154,1155,1156,1157,1158,1159,1160,1161,1162,1163,1164,1165,1166,1167,1168,1169,1170,1171,1172,1173,1174,1175,1176,1177,1178,1179,1180,1181,1182,1183,1184,1185,1186,1187,1188,1189,1190,1191,1192,1193,1194,1195,1196,1197,1198,1199,1200])
channels_idx_bad = np.concatenate((OFFchannels-1, lfpchannels-1, refchannels-1, OUTchannels-1),axis=0)
channels_idx_ON = (np.array([x for x in np.arange(0,1440) if x not in channels_idx_bad]))


#5 seconds traces
time_samples = 100000.0
index1 = np.int(raw_data.shape[1]/10*7)
index2 = np.int(index1 + time_samples)


#Noise calculation
noise_median = np.median(np.abs(raw_data[channels_idx_ON,index1:index2])/0.6745, axis=1)
noise_median_average = np.average(noise_median)
noise_median_stdv = stats.sem(noise_median)
print('#------------------------------------------------------')
print('Noise_Median:'+ str(noise_median))
print('Noise_Median_average:'+ str(noise_median_average))
print('Noise_Median_stdv:'+ str(noise_median_stdv))
print('#------------------------------------------------------')

#Save noise values
analysis_folder =r'Z:\n\Neuroseeker Probe Recordings\Neuroseeker_2017_08_08\Analysis'
filename_Median = os.path.join(analysis_folder +  '\\' + time  + 'noise_Median' + '.npy')
np.save(filename_Median, noise_median)



#Noise, ref internal vs external, 1012 electrodes-----------------------------------------------------------------------

#File names
#17h58, 10 active groups, ref internal
time = '17_58_26.bin'
binary_data_filename = r'Z:\labs2\kampff\n\Neuroseeker Probe Recordings\Neuroseeker_2017_08_08\Data'
binary_data_filename= os.path.join(binary_data_filename, time)
#18h26, 10 active groups, ref external
time = r'18_26_30.bin'
binary_data_filename = r'Z:\labs2\kampff\n\Neuroseeker Probe Recordings\Neuroseeker_2017_08_08\Data'
binary_data_filename= os.path.join(binary_data_filename, time)
# In saline, 12 active groups, ref external
time=r'probe1110_20_21_07'
binary_data_filename = r"Z:\labs2\kampff\n\Neuroseeker Probe Recordings\Neuroseeker_2017_11_27_Probe_Testing\1110\probe1110_20_21_07.bin"


#Open Data
number_of_channels_in_binary_file = 1440
sampling_frequency = 20000
raw_data = np.memmap(binary_data_filename, dtype=np.int16, mode='r')
number_of_timepoints_in_raw = int(raw_data.shape[0] / number_of_channels_in_binary_file)
raw_data = np.reshape(raw_data, (number_of_channels_in_binary_file, number_of_timepoints_in_raw), order='F')


#Electrode numbers, 1012 AP traces
#10 regions ON and half of the group 10 is outside of the brain, 1012 AP traces in total
OFFchannels= np.array([1201,1202,1203,1204,1205,1206,1207,1208,1209,1210,1211,1212,1213,1214,1215,1216,1217,1218,1219,1220,1221,1222,1223,1224,1225,1226,1227,1228,1229,1230,1231,1232,1233,1234,1235,1236,1237,1238,1239,1240,1241,1242,1243,1244,1245,1246,1247,1248,1249,1250,1251,1252,1253,1254,1255,1256,1257,1258,1259,1260,1261,1262,1263,1264,1265,1266,1267,1268,1269,1270,1271,1272,1273,1274,1275,1276,1277,1278,1279,1280,1281,1282,1283,1284,1285,1286,1287,1288,1289,1290,1291,1292,1293,1294,1295,1296,1297,1298,1299,1300,1301,1302,1303,1304,1305,1306,1307,1308,1309,1310,1311,1312,1313,1314,1315,1316,1317,1318,1319,1320,1321,1322,1323,1324,1325,1326,1327,1328,1329,1330,1331,1332,1333,1334,1335,1336,1337,1338,1339,1340,1341,1342,1343,1344,1345,1346,1347,1348,1349,1350,1351,1352,1353,1354,1355,1356,1357,1358,1359,1360,1361,1362,1363,1364,1365,1366,1367,1368,1369,1370,1371,1372,1373,1374,1375,1376,1377,1378,1379,1380,1381,1382,1383,1384,1385,1386,1387,1388,1389,1390,1391,1392,1393,1394,1395,1396,1397,1398,1399,1400,1401,1402,1403,1404,1405,1406,1407,1408,1409,1410,1411,1412,1413,1414,1415,1416,1417,1418,1419,1420,1421,1422,1423,1424,1425,1426,1427,1428,1429,1430,1431,1432,1433,1434,1435,1436,1437,1438,1439,1440])
lfpchannels= np.array([10,30,50,70,90,110,130,150,170,190,210,230,250,270,290,310,330,350,370,390,410,430,450,470,490,510,530,550,570,590,610,630,650,670,690,710,730,750,770,790,810,830,850,870,890,910,930,950,970,990,1010,1030,1050,1070,1090,1110,1130,1150,1170,1190,1210,1230,1250,1270,1290,1310,1330,1350,1370,1390,1410,1430])
refchannels= np.array([57,58,59,60,61,62,63,64,177,178,179,180,181,182,183,184,297,298,299,300,301,302,303,304,417,418,419,420,421,422,423,424,537,538,539,540,541,542,543,544,657,658,659,660,661,662,663,664,777,778,779,780,781,782,783,784,897,898,899,900,901,902,903,904,1017,1018,1019,1020,1021,1022,1023,1024,1137,1138,1139,1140,1141,1142,1143,1144,1257,1258,1259,1260,1261,1262,1263,1264,1377,1378,1379,1380,1381,1382,1383,1384])
#Half of Group 10 OUT
OUTchannels= np.arange(1150, 1200)
channels_idx_bad = np.concatenate((OFFchannels-1, lfpchannels-1, refchannels-1, OUTchannels),axis=0)
#channels_idx_bad = np.concatenate((lfpchannels-1, refchannels-1),axis=0)
channels_idx_ON = (np.array([x for x in np.arange(0,1440) if x not in channels_idx_bad]))


#5 seconds traces
time_samples = 100000.0
index1 = np.int(raw_data.shape[1]/10*7)
index2 = np.int(index1 + time_samples)


#Noise calculation
noise_median = np.median(np.abs(raw_data[channels_idx_ON,index1:index2])/0.6745, axis=1)
noise_median_average = np.average(noise_median)
noise_median_stdv = stats.sem(noise_median)
print('#------------------------------------------------------')
print('Noise_Median:'+ str(noise_median))
print('Noise_Median_average:'+ str(noise_median_average))
print('Noise_Median_stdv:'+ str(noise_median_stdv))
print('#------------------------------------------------------')


#Save noise values
analysis_folder =r'Z:\n\Neuroseeker Probe Recordings\Neuroseeker_2017_08_08\Analysis'
filename_Median = os.path.join(analysis_folder +  '\\' + time  + 'noise_MedianwithGOODchannels_1277ch_from12activeregions' + '.npy')
np.save(filename_Median, noise_median)



#Noise 128ch probe------------------------------------------------------------------------------------------------------

#File names
#noise saline 17_11_17
raw_data_file_ivm= r"S:\swc\kampff\j\Joana Neto\Neuroseeker paper\128chProbe\SalineNoise_20_11_17\noise128ch_saaline2017-11-20T10_34_45.bin"
analysis_folder = r'Z:\n\Neuroseeker Probe Recordings\Neuroseeker_2017_11_27_Probe_Testing\noise17_11_17\noiseaftertrpsina'

#CCU data brain 2015-08-28
analysis_folder = r'Z:\j\Joana Neto\Backup_2017_28_06\PCDisk\Paper Impedance\128chNeuroseeker\CCU\2015_08_28\pair2.2'
#raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2015-08-28T20_15_45.bin')
#raw_data_file_ivm = r"Z:\j\Joana Neto\Backup_2017_28_06\PCDisk\Data_128ch\2015-08-28\Data\amplifier2015-08-28T23_28_20.bin"
raw_data_file_ivm=r"S:\swc\kampff\j\Joana Neto\Backup_2017_28_06\PCDisk\dataset online\Validating 128ch\2015_08_28_Pair_2_0\2015_08_28_Pair_2.2\amplifier2015-08-28T20_15_45.bin"

#Open data
amp_dtype = np.uint16
Probe_y_digitization = 32768
num_ivm_channels = 128
sampling_freq = 30000
filtered_data_type = np.float64
voltage_step_size = 0.195e-6
scale_uV = 1000000
raw_data_ivm = ephys.load_raw_data(raw_data_file_ivm, numchannels=num_ivm_channels, dtype=amp_dtype)
temp_unfiltered = raw_data_ivm.dataMatrix
temp_unfiltered = temp_unfiltered.astype(filtered_data_type)
temp_unfiltered_uV = (temp_unfiltered - Probe_y_digitization) * scale_uV * voltage_step_size

#High-pass filter
high_pass_freq = 500
def highpass(data,BUTTER_ORDER=3, F_HIGH=14250,sampleFreq=30000.0,passFreq=100.0):
    b, a = signal.butter(BUTTER_ORDER,(passFreq/(sampleFreq/2), F_HIGH/(sampleFreq/2)),'pass')
    return signal.filtfilt(b,a,data)
#tfinal = np.ceil((temp_unfiltered_uV.shape[1])/2)
tfinal= 100000
#temp_filtered_uV = highpass(temp_unfiltered_uV[:,0:tfinal], F_HIGH = (sampling_freq/2)*0.95, sampleFreq = sampling_freq, passFreq = high_pass_freq)
temp_filtered_uV = highpass(temp_unfiltered_uV[:,0:tfinal], F_HIGH = (sampling_freq/2)*0.95, sampleFreq = sampling_freq, passFreq = high_pass_freq)

#Noise calculation
noise_median = np.median(np.abs(temp_filtered_uV)/0.6745, axis=1)
noise_median_average = np.average(noise_median)
noise_median_stdv = stats.sem(noise_median)
print('#------------------------------------------------------')
print('Noise_Median:'+ str(noise_median))
print('Noise_Median_average:'+ str(noise_median_average))
print('Noise_Median_stdv:'+ str(noise_median_stdv))
print('#------------------------------------------------------')


#Save noise values
analysis_folder=r'C:\Users\KAMPFF-LAB-ANALYSIS3\Google Drive\Thesis Chapter\Chapter 4\Pictures\Previous_figure3_ch3\Noise\Previous\Noise 128chprobe\Noise_brain\amplifier2015-08-28T20_15_45\pair2.2'
filename_Median = os.path.join(analysis_folder + '\\' + str(high_pass_freq) + 'noise_Median' + '.npy')
np.save(filename_Median, noise_median)







































#code not in use -------------------------------------------------------------------------------------------------------

#after REF each group and highpass
time ='18_26_30_afterREFeachGroup.bin'
binary_data_filename = r'Z:\n\Neuroseeker Probe Recordings\Neuroseeker_2017_08_08\Data'
binary_data_filename= os.path.join(binary_data_filename, time)
binary_data_filename=r"Z:\n\Neuroseeker Probe Recordings\Neuroseeker_2017_08_08\Data\18_26_30_afterReFeachGroup_whighpass.bin"

#after REF each group
time = r'18_26_30_afterREFeachGroupwoutHighpass.bin'
binary_data_filename = r'Z:\n\Neuroseeker Probe Recordings\Neuroseeker_2017_08_08\Data'
binary_data_filename= os.path.join(binary_data_filename, time)

#19h52#12active
binary_data_filename =r'F:\chapter 2\Neuroseeker_2017_08_08\Data\19_52_55.bin'

#6active
binary_data_filename = r'F:\chapter 2\Neuroseeker_2017_08_08\Data\20_13_45.bin'
binary_data_filename = r'F:\chapter 2\Neuroseeker_2017_08_08\Data'
time = r'20_19_15.bin'
binary_data_filename= os.path.join(binary_data_filename, time)

#----------------




    # NOISE MEDIAN for each group -------------------------------------------------------------------------------------------
    # Protocol 1 to calculate the stdv from noise MEDIAN
    # 5 seconds traces
    time_samples = 100000.0
    index1 = np.int(raw_data.shape[1] / 10 * 7)
    index2 = np.int(index1 + time_samples)
    groups = np.arange(1, 13)
    for g in groups:
        AP = np.arange((g - 1) * 120, g * 120)
        noise_median = np.median(np.abs(raw_data[AP, index1:index2]) / 0.6745, axis=1)
        noise_median_average = np.average(noise_median)
        noise_median_stdv = stats.sem(noise_median)
        print('#------------------------------------------------------')
        print('GroupNumber:' + str(g))
        print('Noise_Median:' + str(noise_median))
        print('Noise_Median_average:' + str(noise_median_average))
        print('Noise_Median_stdv:' + str(noise_median_stdv))
        print('#------------------------------------------------------')

    # NOISE MEDIAN for all groups together-----------------------------------------------------------------------------------
    noise_median = np.median(np.abs(raw_data[:, index1:index2]) / 0.6745, axis=1)
    noise_median_average = np.average(noise_median)
    noise_median_stdv = stats.sem(noise_median)
    print('#------------------------------------------------------')
    print('Noise_Median:' + str(noise_median))
    print('Noise_Median_average:' + str(noise_median_average))
    print('Noise_Median_stdv:' + str(noise_median_stdv))
    print('#------------------------------------------------------')

    # analysis_folder =r'Z:\j\Neuroseeker256ch\Type1_Probe6\2017-02-08\Data\Noise saline'
    # analysis_folder =r'Z:\j\Neuroseeker256ch\Type1_Probe6\2017-02-08\Datakilosort\Nfilt_Test\nfilt512\amplifier2017-02-08T15_34_04'
    analysis_folder = r'Z:\n\Neuroseeker Probe Recordings\Neuroseeker_2017_08_08\Analysis'
    filename_Median = os.path.join(analysis_folder + '\\' + time + 'noise_MedianAllGroups_1440chs' + '.npy')
    np.save(filename_Median, noise_median)


#External reference data
#File names
#Calculation of the noise for 2 regions
#2regionsON
time = r'18_18_41.bin'
binary_data_filename=r"Z:\n\Neuroseeker Probe Recordings\Neuroseeker_2017_08_08\Data\18_18_41.bin"
#4regionsON
time = r'18_20_31.bin'
binary_data_filename=r"Z:\n\Neuroseeker Probe Recordings\Neuroseeker_2017_08_08\Data\18_20_31.bin"
#6regions ON
time =r'18_22_40.bin'
binary_data_filename = r"Z:\n\Neuroseeker Probe Recordings\Neuroseeker_2017_08_08\Data\18_22_40.bin"
#8regions active
time= r'18_24_20.bin'
binary_data_filename =r"Z:\n\Neuroseeker Probe Recordings\Neuroseeker_2017_08_08\Data\18_24_20.bin"
#10regions
binary_data_filename = r"Z:\n\Neuroseeker Probe Recordings\Neuroseeker_2017_08_08\Data\18_26_30.bin"
time=r'18_26_30.bin'
#12active regions
time =r'18_40_36.bin'
binary_data_filename = r'Z:\n\Neuroseeker Probe Recordings\Neuroseeker_2017_08_08\Data'
binary_data_filename= os.path.join(binary_data_filename, time)


#Open data
number_of_channels_in_binary_file = 1440
sampling_frequency = 20000
raw_data = np.memmap(binary_data_filename, dtype=np.int16, mode='r')
number_of_timepoints_in_raw = int(raw_data.shape[0] / number_of_channels_in_binary_file)
raw_data = np.reshape(raw_data, (number_of_channels_in_binary_file, number_of_timepoints_in_raw), order='F')


#Electrode numbers, 212 AP traces
OFFchannels= np.arange(241,1441)
lfpchannels= np.array([10,30,50,70,90,110,130,150,170,190,210,230,250,270,290,310,330,350,370,390,410,430,450,470,490,510,530,550,570,590,610,630,650,670,690,710,730,750,770,790,810,830,850,870,890,910,930,950,970,990,1010,1030,1050,1070,1090,1110,1130,1150,1170,1190,1210,1230,1250,1270,1290,1310,1330,1350,1370,1390,1410,1430])
refchannels= np.array([57,58,59,60,61,62,63,64,177,178,179,180,181,182,183,184,297,298,299,300,301,302,303,304,417,418,419,420,421,422,423,424,537,538,539,540,541,542,543,544,657,658,659,660,661,662,663,664,777,778,779,780,781,782,783,784,897,898,899,900,901,902,903,904,1017,1018,1019,1020,1021,1022,1023,1024,1137,1138,1139,1140,1141,1142,1143,1144,1257,1258,1259,1260,1261,1262,1263,1264,1377,1378,1379,1380,1381,1382,1383,1384])
OUTchannels= np.array([1081,1082,1083,1084,1085,1086,1087,1088,1089,1090,1091,1092,1093,1094,1095,1096,1097,1098,1099,1100,1101,1102,1103,1104,1105,1106,1107,1108,1109,1110,1111,1112,1113,1114,1115,1116,1117,1118,1119,1120,1121,1122,1123,1124,1125,1126,1127,1128,1129,1130,1131,1132,1133,1134,1135,1136,1137,1138,1139,1140,1141,1142,1143,1144,1145,1146,1147,1148,1149,1150,1151,1152,1153,1154,1155,1156,1157,1158,1159,1160,1161,1162,1163,1164,1165,1166,1167,1168,1169,1170,1171,1172,1173,1174,1175,1176,1177,1178,1179,1180,1181,1182,1183,1184,1185,1186,1187,1188,1189,1190,1191,1192,1193,1194,1195,1196,1197,1198,1199,1200])
channels_idx_bad = np.concatenate((OFFchannels-1, lfpchannels-1, refchannels-1, OUTchannels),axis=0)
channels_idx_ON = (np.array([x for x in np.arange(0,1440) if x not in channels_idx_bad]))


#5 seconds traces
time_samples = 100000.0
index1 = np.int(raw_data.shape[1]/10*7)
index2 = np.int(index1 + time_samples)


#Noise calculation
noise_median = np.median(np.abs(raw_data[channels_idx_ON,index1:index2])/0.6745, axis=1)
noise_median_average = np.average(noise_median)
noise_median_stdv = stats.sem(noise_median)
print('#------------------------------------------------------')
print('Noise_Median:'+ str(noise_median))
print('Noise_Median_average:'+ str(noise_median_average))
print('Noise_Median_stdv:'+ str(noise_median_stdv))
print('#------------------------------------------------------')


#Save noise values
analysis_folder =r'Z:\n\Neuroseeker Probe Recordings\Neuroseeker_2017_08_08\Analysis'
filename_Median = os.path.join(analysis_folder +  '\\' + time  + 'noise_Median' + '.npy')
np.save(filename_Median, noise_median)

#File names


#Open Data
number_of_channels_in_binary_file = 1440
sampling_frequency = 20000
raw_data = np.memmap(binary_data_filename, dtype=np.int16, mode='r')
number_of_timepoints_in_raw = int(raw_data.shape[0] / number_of_channels_in_binary_file)
raw_data = np.reshape(raw_data, (number_of_channels_in_binary_file, number_of_timepoints_in_raw), order='F')


#Electrode numbers, 1012 AP traces
#10 regions ON and half
OFFchannels= np.array([1201,1202,1203,1204,1205,1206,1207,1208,1209,1210,1211,1212,1213,1214,1215,1216,1217,1218,1219,1220,1221,1222,1223,1224,1225,1226,1227,1228,1229,1230,1231,1232,1233,1234,1235,1236,1237,1238,1239,1240,1241,1242,1243,1244,1245,1246,1247,1248,1249,1250,1251,1252,1253,1254,1255,1256,1257,1258,1259,1260,1261,1262,1263,1264,1265,1266,1267,1268,1269,1270,1271,1272,1273,1274,1275,1276,1277,1278,1279,1280,1281,1282,1283,1284,1285,1286,1287,1288,1289,1290,1291,1292,1293,1294,1295,1296,1297,1298,1299,1300,1301,1302,1303,1304,1305,1306,1307,1308,1309,1310,1311,1312,1313,1314,1315,1316,1317,1318,1319,1320,1321,1322,1323,1324,1325,1326,1327,1328,1329,1330,1331,1332,1333,1334,1335,1336,1337,1338,1339,1340,1341,1342,1343,1344,1345,1346,1347,1348,1349,1350,1351,1352,1353,1354,1355,1356,1357,1358,1359,1360,1361,1362,1363,1364,1365,1366,1367,1368,1369,1370,1371,1372,1373,1374,1375,1376,1377,1378,1379,1380,1381,1382,1383,1384,1385,1386,1387,1388,1389,1390,1391,1392,1393,1394,1395,1396,1397,1398,1399,1400,1401,1402,1403,1404,1405,1406,1407,1408,1409,1410,1411,1412,1413,1414,1415,1416,1417,1418,1419,1420,1421,1422,1423,1424,1425,1426,1427,1428,1429,1430,1431,1432,1433,1434,1435,1436,1437,1438,1439,1440])
lfpchannels= np.array([10,30,50,70,90,110,130,150,170,190,210,230,250,270,290,310,330,350,370,390,410,430,450,470,490,510,530,550,570,590,610,630,650,670,690,710,730,750,770,790,810,830,850,870,890,910,930,950,970,990,1010,1030,1050,1070,1090,1110,1130,1150,1170,1190,1210,1230,1250,1270,1290,1310,1330,1350,1370,1390,1410,1430])
refchannels= np.array([57,58,59,60,61,62,63,64,177,178,179,180,181,182,183,184,297,298,299,300,301,302,303,304,417,418,419,420,421,422,423,424,537,538,539,540,541,542,543,544,657,658,659,660,661,662,663,664,777,778,779,780,781,782,783,784,897,898,899,900,901,902,903,904,1017,1018,1019,1020,1021,1022,1023,1024,1137,1138,1139,1140,1141,1142,1143,1144,1257,1258,1259,1260,1261,1262,1263,1264,1377,1378,1379,1380,1381,1382,1383,1384])
OUTchannels= np.arange(1150, 1200)
channels_idx_bad = np.concatenate((OFFchannels-1, lfpchannels-1, refchannels-1, OUTchannels),axis=0)
channels_idx_ON = (np.array([x for x in np.arange(0,1440) if x not in channels_idx_bad]))


#Calculation noise
noise_median = np.median(np.abs(raw_data[channels_idx_ON,index1:index2])/0.6745, axis=1)
noise_median_average = np.average(noise_median)
noise_median_stdv = stats.sem(noise_median)
print('#------------------------------------------------------')
print('Noise_Median:'+ str(noise_median))
print('Noise_Median_average:'+ str(noise_median_average))
print('Noise_Median_stdv:'+ str(noise_median_stdv))
print('#------------------------------------------------------')


#Save noise values
analysis_folder =r'Z:\n\Neuroseeker Probe Recordings\Neuroseeker_2017_08_08\Analysis'
filename_Median = os.path.join(analysis_folder +  '\\' + time  + 'noise_Median' + '.npy')
np.save(filename_Median, noise_median)
#analysis_folder=r'Z:\n\Neuroseeker Probe Recordings\Neuroseeker_2017_11_27_Probe_Testing\1110'




#Protocol 2 to calculate the stdv from noise RMS

# RMS noise level for all channels
def RMS_calculation(data):

    NumSites = 1440
    RMS = np.zeros(NumSites)

    for i in range(NumSites):
        RMS[i] = np.sqrt((1/len(data[i]))*np.sum(data[i]**2))

    return RMS

noise_rms = RMS_calculation(temp_filtered_uV)
noise_rms_average = np.average(noise_rms)
noise_rms_stdv = stats.sem(noise_rms)

print('#------------------------------------------------------')
print('RMS:'+ str(noise_rms))
print('RMS_average:'+ str(noise_rms_average))
print('RMS_average_stdv:'+ str(noise_rms_stdv))
print('#------------------------------------------------------')

analysis_folder =r'Z:\j\Neuroseeker256ch\Type1_Probe6\2017-02-08\Datakilosort\Nfilt_Test\nfilt512'

filename_RMS = os.path.join(analysis_folder + '\\' + str(high_pass_freq) + 'noise_RMS' + '.npy')

np.save(filename_RMS, noise_rms)


analysis_folder =r'Z:\j\Neuroseeker256ch\Type1_Probe6\2017-02-08\Data\Noise saline'
filename_RMS = os.path.join(analysis_folder + '\\' + 'noise_RMS' + '.npy')
np.save(filename_RMS, noise_rms)


#18h26
#raw data
#binary_data_filename =r'F:\chapter 2\Neuroseeker_2017_08_08\Data\18_26_30.bin'
#after REF each group and highpass
#time ='18_26_30_afterREFeachGroup.bin'
#binary_data_filename = r'Z:\n\Neuroseeker Probe Recordings\Neuroseeker_2017_08_08\Data'
#binary_data_filename= os.path.join(binary_data_filename, time)
#after REF each group wout Highpass
#time = r'18_26_30_afterREFeachGroupwoutHighpass.bin'
#binary_data_filename = r'Z:\n\Neuroseeker Probe Recordings\Neuroseeker_2017_08_08\Data'
#binary_data_filename= os.path.join(binary_data_filename, time)



#plot noises of increasing groups on
#18h26

#g2 = np.load(r"Z:\n\Neuroseeker Probe Recordings\Neuroseeker_2017_08_08\Analysis\Noise_Median_2groups\EXT\18_18_41.binnoise_Median.npy")
#g4 = np.load(r"Z:\n\Neuroseeker Probe Recordings\Neuroseeker_2017_08_08\Analysis\Noise_Median_2groups\EXT\18_20_31.binnoise_Median.npy")
#g6 = np.load(r"Z:\n\Neuroseeker Probe Recordings\Neuroseeker_2017_08_08\Analysis\Noise_Median_2groups\EXT\18_22_40.binnoise_Median.npy")
#g8 = np.load(r"Z:\n\Neuroseeker Probe Recordings\Neuroseeker_2017_08_08\Analysis\Noise_Median_2groups\EXT\18_24_20.binnoise_Median.npy")
#g10 = np.load(r"Z:\n\Neuroseeker Probe Recordings\Neuroseeker_2017_08_08\Analysis\Noise_Median_2groups\EXT\18_26_30_afterREFeachGroupwoutHighpass.binnoise_Median.npy")
#g12 = np.load(r"Z:\n\Neuroseeker Probe Recordings\Neuroseeker_2017_08_08\Analysis\Noise_Median_2groups\EXT\18_40_36.binnoise_Median.npy")


g2 = np.load(r"Z:\n\Neuroseeker Probe Recordings\Neuroseeker_2017_08_08\Analysis\Noise_Median_2groups\EXT\18_18_41_afterReFeachGroup_whighpassnoise_Median.npy")
g4 = np.load(r"Z:\n\Neuroseeker Probe Recordings\Neuroseeker_2017_08_08\Analysis\Noise_Median_2groups\EXT\18_20_31_afterReFeachGroup_whighpassnoise_Median.npy")
g6 = np.load(r"Z:\n\Neuroseeker Probe Recordings\Neuroseeker_2017_08_08\Analysis\Noise_Median_2groups\EXT\18_22_40_afterReFeachGroup_whighpassnoise_Median.npy")
g8 = np.load(r"Z:\n\Neuroseeker Probe Recordings\Neuroseeker_2017_08_08\Analysis\Noise_Median_2groups\EXT\18_24_20_afterReFeachGroup_whighpassnoise_Median.npy")
g10 = np.load(r"Z:\n\Neuroseeker Probe Recordings\Neuroseeker_2017_08_08\Analysis\Noise_Median_2groups\EXT\18_26_30_afterReFeachGroup_whighpassnoise_Median.npy")
g12 = np.load(r"Z:\n\Neuroseeker Probe Recordings\Neuroseeker_2017_08_08\Analysis\Noise_Median_2groups\EXT\18_40_36_afterReFeachGroup_whighpassnoise_Median.npy")




noise = np.asarray((g2,g4,g6,g8,g10,g12))
data = noise.T


plt.figure()

plt.boxplot(data, showmeans=True)

f, ax = plt.subplots()
sns.violinplot(data=data)
sns.despine(offset=10, trim=True)

plt.figure()

sns.boxplot(data=data)

sns.swarmplot(data=data, color=".25")

#compare diferent files
#noise1= np.load(r"Z:\n\Neuroseeker Probe Recordings\Neuroseeker_2017_08_08\Analysis\18_26\18_26_30_afterREFeachGroup.binnoise_MedianwithGOODchannels_1012ch_from10activeregions.npy")

#noise2= np.load(r"Z:\n\Neuroseeker Probe Recordings\Neuroseeker_2017_08_08\Analysis\18_26\18_26_30_afterREFeachGroupwoutHighpass.binnoise_MedianwithGOODchannels_1012ch_from10activeregions.npy")

#noise3= np.load(r"Z:\n\Neuroseeker Probe Recordings\Neuroseeker_2017_08_08\Analysis\17_58\17_58_26.binnoise_MedianwithGOODchannels_1012ch_from10activeregions.npy")



noise2= np.load(r"Z:\n\Neuroseeker Probe Recordings\Neuroseeker_2017_08_08\Analysis\18_26\18_26_30_afterReFeachGroup_whighpassnoise_Median.npy")

noise3= np.load(r"Z:\n\Neuroseeker Probe Recordings\Neuroseeker_2017_08_08\Analysis\17_58\17_58_26_afterhighpassnoise_MedianwithGOODchannels_1012ch_from10activeregions.npy")



#noise = np.asarray((noise1,noise2,noise3))

noise = np.asarray((noise2,noise3))
data = noise.T

plt.figure()
f, ax = plt.subplots()
sns.violinplot(data=data)
sns.despine(offset=10, trim=True)


plt.figure()
f, ax = plt.subplots()
sns.boxplot(data=data)
#plt.ylim(0,35)
sns.despine(offset=10, trim=True)

plt.figure()
sns.boxplot(data=noise3)
plt.ylim(0,35)


#XXXXXXXXXXLowpass and Highpass signal


filtered_data_type = np.float64
#iir_params = {'order': 3, 'ftype': 'butter', 'padlen': 0}
#low_pass_freq = 3000.0
high_pass_freq = 500.0
offset_microvolt = 200



def highpass(data,BUTTER_ORDER=3, F_HIGH=14250,sampleFreq=20000.0,passFreq=100.0):
    b, a = signal.butter(BUTTER_ORDER,(passFreq/(sampleFreq/2), F_HIGH/(sampleFreq/2)),'pass')
    return signal.filtfilt(b,a,data)


plt.figure()
for i in np.arange(0, np.shape(AP)[0]):
    temp_unfiltered = (raw_data[AP[i], index1:index2]).astype(filtered_data_type)
    temp_unfiltered = temp_unfiltered.astype(filtered_data_type)
    temp_filtered = highpass(temp_unfiltered, F_HIGH=(sampling_frequency / 2) * 0.95,
                             sampleFreq=sampling_frequency, passFreq=high_pass_freq)
    #temp_filtered = filters.low_pass_filter(temp_filtered_high, sampling_frequency, low_pass_freq, method='iir', iir_params=iir_params)
    plt.plot(temp_filtered.T + i*offset_microvolt)
    plt.title(np.str(AP))
    plt.show()

#vertical probe

#chronic probe
channels_idx_bad= np.asanyarray([9,29,49,56,57,58,59,60,61,62,63,69,89,109,129,149,169,176,177,178,179,180,181,182,183,189,209,229,249,269,289,296,297,298,299,300,301,302,303,309,329,349,369,389,409,416,417,418,419,420,421,422,423,429,449,469,489,509,529,536,537,538,539,540,541,542,543,549,569,589,609,629,649,656,657,658,659,660,661,662,663,669,689,709,720,721,722,723,724,725,726,727,728,729,730,731,732,733,734,735,736,737,738,739,740,741,742,743,744,745,746,747,748,749,750,751,752,753,754,755,756,757,758,759,769,776,777,778,779,780,781,782,783,789,809,829,849,869,889,896,897,898,899,900,901,902,903,909,929,949,969,989,1009,1016,1017,1018,1019,1020,1021,1022,1023,1029,1049,1069,1089,1109,1129,1136,1137,1138,1139,1140,1141,1142,1143,1149,1169,1189,1200,1201,1202,1203,1204,1205,1206,1207,1208,1209,1210,1211,1212,1213,1214,1215,1216,1217,1218,1219,1220,1221,1222,1223,1224,1225,1226,1227,1228,1229,1230,1231,1232,1233,1234,1235,1236,1237,1238,1239,1249,1256,1257,1258,1259,1260,1261,1262,1263,1269,1289,1309,1329,1349,1369,1376,1377,1378,1379,1380,1381,1382,1383,1389,1409,1429])
channels_idx_ON = (np.array([x for x in np.arange(0,1440) if x not in channels_idx_bad]))



#angle probe
#binary_data_filename = r'Z:\n\Neuroseeker Probe Recordings\Neuroseeker_2017_03_28_Anesthesia_Auditory_DoubleProbes\Angled\Data\BinFormat\Exp2and3_2017_03_28T18_48_25_Amp_S16_LP3p5KHz_mV_REFsubmatrix_HP600_1440ch.bin'
#binary_data_filename = r'Z:\n\Neuroseeker Probe Recordings\Neuroseeker_2017_03_28_Anesthesia_Auditory_DoubleProbes\Angled\Data\Exp2and3_2017_03_28T18_48_25_Amp_S16_LP3p5KHz_LFPs_100hz_mV.bin' # Desktop
binary_data_filename = r'Z:\n\Neuroseeker Probe Recordings\Neuroseeker_2017_03_28_Anesthesia_Auditory_DoubleProbes\Angled\Analysis\Experiment_2_T18_48_25_And_Experiment_3_T19_41_07\Kilosort\Exp2and3_2017_03_28T18_48_25_Amp_S16_LP3p5KHz_mV.bin'

#anotherrecording
#binary_data_filename = r'F:\chapter 2\Chronic\day2017-05-22\InitialTest\test_1_2017-05-22T15_29_08\test_1_2017-05-22T15_29_08_Amp_S16_LP3p5KHz_mV.bin'
binary_data_filename = r'F:\chapter 2\2017_10_26_aulaphdstudents\rec10_Amp_S16_LP3p5KHz_mV.bin'
binary_data_filename =r'F:\chapter 2\Chronic\day2017-05-22\InitialTest\test_1_2017-05-22T15_29_08\test_1_2017-05-22T15_29_08_Amp_S16_LP3p5KHz_wreferncech_mV.bin'

#channels_OFF_BAD_LFPS_REF
badchannels= np.array([721,722,723,724,725,726,727,728,729,730,731,732,733,734,735,736,737,738,739,740,741,742,743,744,745,746,747,748,749,750,751,752,753,754,755,756,757,758,759,760,761,762,763,764,765,766,767,768])
OFFchannels= np.array([1201,1202,1203,1204,1205,1206,1207,1208,1209,1210,1211,1212,1213,1214,1215,1216,1217,1218,1219,1220,1221,1222,1223,1224,1225,1226,1227,1228,1229,1230,1231,1232,1233,1234,1235,1236,1237,1238,1239,1240,1241,1242,1243,1244,1245,1246,1247,1248,1249,1250,1251,1252,1253,1254,1255,1256,1257,1258,1259,1260,1261,1262,1263,1264,1265,1266,1267,1268,1269,1270,1271,1272,1273,1274,1275,1276,1277,1278,1279,1280,1281,1282,1283,1284,1285,1286,1287,1288,1289,1290,1291,1292,1293,1294,1295,1296,1297,1298,1299,1300,1301,1302,1303,1304,1305,1306,1307,1308,1309,1310,1311,1312,1313,1314,1315,1316,1317,1318,1319,1320,1321,1322,1323,1324,1325,1326,1327,1328,1329,1330,1331,1332,1333,1334,1335,1336,1337,1338,1339,1340,1341,1342,1343,1344,1345,1346,1347,1348,1349,1350,1351,1352,1353,1354,1355,1356,1357,1358,1359,1360,1361,1362,1363,1364,1365,1366,1367,1368,1369,1370,1371,1372,1373,1374,1375,1376,1377,1378,1379,1380,1381,1382,1383,1384,1385,1386,1387,1388,1389,1390,1391,1392,1393,1394,1395,1396,1397,1398,1399,1400,1401,1402,1403,1404,1405,1406,1407,1408,1409,1410,1411,1412,1413,1414,1415,1416,1417,1418,1419,1420,1421,1422,1423,1424,1425,1426,1427,1428,1429,1430,1431,1432,1433,1434,1435,1436,1437,1438,1439,1440])
lfpchannels= np.array([10,30,50,70,90,110,130,150,170,190,210,230,250,270,290,310,330,350,370,390,410,430,450,470,490,510,530,550,570,590,610,630,650,670,690,710,730,750,770,790,810,830,850,870,890,910,930,950,970,990,1010,1030,1050,1070,1090,1110,1130,1150,1170,1190,1210,1230,1250,1270,1290,1310,1330,1350,1370,1390,1410,1430])
refchannels= np.array([57,58,59,60,61,62,63,64,177,178,179,180,181,182,183,184,297,298,299,300,301,302,303,304,417,418,419,420,421,422,423,424,537,538,539,540,541,542,543,544,657,658,659,660,661,662,663,664,777,778,779,780,781,782,783,784,897,898,899,900,901,902,903,904,1017,1018,1019,1020,1021,1022,1023,1024,1137,1138,1139,1140,1141,1142,1143,1144,1257,1258,1259,1260,1261,1262,1263,1264,1377,1378,1379,1380,1381,1382,1383,1384])
channels_idx_bad = np.concatenate((badchannels-1,OFFchannels-1,lfpchannels-1, refchannels),axis=0)
channels_idx_ON = (np.array([x for x in np.arange(0,1440) if x not in all_channels_bad]))
np.savetxt(os.path.join(r'C:\Users\KAMPFF-LAB-ANALYSIS3\Desktop\chapter 2\Neuroseeker_2017_03_28_Anesthesia_Auditory_DoubleProbes\Angled', "channels_ON.txt"), channels_idx_ON, fmt='%d', delimiter=',', newline=',')



#plot APs from each region for the angle

APSN= [18, 19, 20, 21, 22, 23]
APNucleus= [355,356,357,358,359,360]
#APDG= [799,800,801,802,803,804]
APDG= [457,458,459,460,461,462]
#APCA=[913,914,915,916,917,918]
APCA= [799,800,801,802,803,804]
#APcortex= [1070,1071,1072,1073,1074,1075]
APcortex= [913,914,915,916,917,918, 1070,1071,1072,1073,1074,1075]



#APcortex= [ 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025,
#       1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036,
#       1037, 1038, 1039, 1040, 1041, 1042, 1043, 1044, 1045, 1046, 1047,
#       1048, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058,
#       1059, 1060, 1061, 1062, 1063, 1064, 1065, 1066, 1067, 1068, 1069,
#       1070, 1071, 1072, 1073, 1074, 1075, 1076, 1077, 107, 1079, 1080, 1081, 1082, 1083, 1084]

#APcortex = [1180, 1181, 1182, 1183, 1184, 1185]
#APcortex =[1164, 1165, 1166, 1167, 1168]
#spikes in SN in ch 200, 201


AP = np.concatenate((APSN, APNucleus, APDG, APCA, APcortex), axis=0)


APg11= np.arange(1200,1319)

AP=APg11
time_samples = 1000000.0
index1 = np.int(raw_data.shape[1]/10*7)
index2 = np.int(index1 + time_samples)


plt.figure()
offset_microvolt= 200
for i in np.arange(0, np.shape(AP)[0]):
    plt.plot(raw_data[AP[i], index1:index2].T + i*offset_microvolt)
    plt.title(np.str((AP, index1,index2)))
    plt.show()





lfpsSN = [9,   29,   49,   69,   89,  109]
#lfpsNucleus = [369,  389,  409,  429, 449,  469]
lfpsNucleus = [249,  269,  289,  309,  329,  349,  369,  389]

#lfpsCA1= [849, 869, 889, 909, 929, 949]
lfpsCA1= [709,  769,  789,  809,  829,  849]
#lfpsDG = [689,  709,  769,  789,  809,  829]
lfpsDG = [469,  489,  509,  529,  549,  569,  589,  609,  629,  649,669,  689]
lfpsCORTEXsuperficial = [1089, 1109, 1129, 1149, 1169, 1189]
lfpsCORTEXdeep = [969, 989, 1009, 1029, 1049, 1069]
lfpsCORTEX =[969, 989, 1009, 1029, 1049, 1069, 1129, 1149, 1169, 1189]

lfp = np.concatenate((lfpsSN, lfpsNucleus,lfpsDG,lfpsCA1,lfpsCORTEX),axis=0)

time_samples = 1000000.0
#index1 = np.int(raw_data.shape[1]/2)
#index2 = np.int(raw_data.shape[1]/2 + time_samples)

index1 = np.int(raw_data.shape[1]/2)
index2 = np.int(index1 + time_samples)
#index1 = 500000
#index2 = 1000000

offset_microvolt = 700
plt.figure()
for i in np.arange(0, np.shape(lfp)[0]):
    plt.plot(raw_data[lfp[i], index1:index2].T + i*offset_microvolt)
    plt.title(np.str((lfp,index1,index2)))
    plt.show()



#10regions ON and half of the group 10  is outside of the brain
OFFchannels= np.array([1201,1202,1203,1204,1205,1206,1207,1208,1209,1210,1211,1212,1213,1214,1215,1216,1217,1218,1219,1220,1221,1222,1223,1224,1225,1226,1227,1228,1229,1230,1231,1232,1233,1234,1235,1236,1237,1238,1239,1240,1241,1242,1243,1244,1245,1246,1247,1248,1249,1250,1251,1252,1253,1254,1255,1256,1257,1258,1259,1260,1261,1262,1263,1264,1265,1266,1267,1268,1269,1270,1271,1272,1273,1274,1275,1276,1277,1278,1279,1280,1281,1282,1283,1284,1285,1286,1287,1288,1289,1290,1291,1292,1293,1294,1295,1296,1297,1298,1299,1300,1301,1302,1303,1304,1305,1306,1307,1308,1309,1310,1311,1312,1313,1314,1315,1316,1317,1318,1319,1320,1321,1322,1323,1324,1325,1326,1327,1328,1329,1330,1331,1332,1333,1334,1335,1336,1337,1338,1339,1340,1341,1342,1343,1344,1345,1346,1347,1348,1349,1350,1351,1352,1353,1354,1355,1356,1357,1358,1359,1360,1361,1362,1363,1364,1365,1366,1367,1368,1369,1370,1371,1372,1373,1374,1375,1376,1377,1378,1379,1380,1381,1382,1383,1384,1385,1386,1387,1388,1389,1390,1391,1392,1393,1394,1395,1396,1397,1398,1399,1400,1401,1402,1403,1404,1405,1406,1407,1408,1409,1410,1411,1412,1413,1414,1415,1416,1417,1418,1419,1420,1421,1422,1423,1424,1425,1426,1427,1428,1429,1430,1431,1432,1433,1434,1435,1436,1437,1438,1439,1440])
lfpchannels= np.array([10,30,50,70,90,110,130,150,170,190,210,230,250,270,290,310,330,350,370,390,410,430,450,470,490,510,530,550,570,590,610,630,650,670,690,710,730,750,770,790,810,830,850,870,890,910,930,950,970,990,1010,1030,1050,1070,1090,1110,1130,1150,1170,1190,1210,1230,1250,1270,1290,1310,1330,1350,1370,1390,1410,1430])
refchannels= np.array([57,58,59,60,61,62,63,64,177,178,179,180,181,182,183,184,297,298,299,300,301,302,303,304,417,418,419,420,421,422,423,424,537,538,539,540,541,542,543,544,657,658,659,660,661,662,663,664,777,778,779,780,781,782,783,784,897,898,899,900,901,902,903,904,1017,1018,1019,1020,1021,1022,1023,1024,1137,1138,1139,1140,1141,1142,1143,1144,1257,1258,1259,1260,1261,1262,1263,1264,1377,1378,1379,1380,1381,1382,1383,1384])
#Group 10 out of the
#OUTchannels= np.array([1081,1082,1083,1084,1085,1086,1087,1088,1089,1090,1091,1092,1093,1094,1095,1096,1097,1098,1099,1100,1101,1102,1103,1104,1105,1106,1107,1108,1109,1110,1111,1112,1113,1114,1115,1116,1117,1118,1119,1120,1121,1122,1123,1124,1125,1126,1127,1128,1129,1130,1131,1132,1133,1134,1135,1136,1137,1138,1139,1140,1141,1142,1143,1144,1145,1146,1147,1148,1149,1150,1151,1152,1153,1154,1155,1156,1157,1158,1159,1160,1161,1162,1163,1164,1165,1166,1167,1168,1169,1170,1171,1172,1173,1174,1175,1176,1177,1178,1179,1180,1181,1182,1183,1184,1185,1186,1187,1188,1189,1190,1191,1192,1193,1194,1195,1196,1197,1198,1199,1200])
#Half of Group10 IN
OUTchannels= np.arange(1150, 1200)
channels_idx_bad = np.concatenate((OFFchannels-1, lfpchannels-1, refchannels-1, OUTchannels),axis=0)
channels_idx_ON = (np.array([x for x in np.arange(0,1440) if x not in channels_idx_bad]))

APThalamus= channels_idx_ON[:106*4]
APhipocampus= channels_idx_ON[106*4: 106*7]
APcortex = channels_idx_ON[106*7: 1012]



AP = np.concatenate((APThalamus,APhipocampus,APcortex), axis=0)

#cortex
AP= np.arange(1080,1199)
AP= np.arange(960,1079)
#AP=np.arange(960,1160)
APC0=np.arange(852,858)
APC1=np.arange(960+26,960+26+7)
APC2=np.arange(1080+15-7,1080+15)
APC3=np.arange(1100,1106)
APC4= np.arange(1150,1153)

#hipocampus

AP= np.arange(480,599)
AP=np.arange(600, 719)
AP = np.arange(720, 839)

APH0=np.arange(480,482)
#APH1=np.arange(551,557)
APH1=np.arange(599-8-12, 599-8)
APH2 =np.arange(690,696)
APH3=np.arange(768,776)
#thalamus
AP4=np.arange(360,479)
AP3= np.arange(240,359)
AP2 = np.arange(120,239)
AP1= np.arange(0,119)

APT1= np.arange(15,15+6)
APT1_1=np.arange(94,99)
APT2= np.arange(137, 149)
APT3_1=np.arange(262,267)
APT3= np.arange(313,319)
APT4=np.arange(366,366+6)
APT5= np.arange(462,466)

#AP= np.concatenate((APC0,APC1,APC2,APC3,APC4),axis=0)
AP= np.concatenate((APT1,APT2,APT3,APT4,APT5,APH0,APH1,APH2,APH3,APC0,APC1,APC2,APC3,APC4),axis=0)

time_samples = 100000
index1 = np.int(raw_data.shape[1]/10*6) - 2000
index2 = np.int(index1 + time_samples)


#time_samples = 200000.0
#index1 = np.int(raw_data.shape[1]/10*9) - 2000
#index2 = np.int(index1 + time_samples)

OFFchannels= np.array([1201,1202,1203,1204,1205,1206,1207,1208,1209,1210,1211,1212,1213,1214,1215,1216,1217,1218,1219,1220,1221,1222,1223,1224,1225,1226,1227,1228,1229,1230,1231,1232,1233,1234,1235,1236,1237,1238,1239,1240,1241,1242,1243,1244,1245,1246,1247,1248,1249,1250,1251,1252,1253,1254,1255,1256,1257,1258,1259,1260,1261,1262,1263,1264,1265,1266,1267,1268,1269,1270,1271,1272,1273,1274,1275,1276,1277,1278,1279,1280,1281,1282,1283,1284,1285,1286,1287,1288,1289,1290,1291,1292,1293,1294,1295,1296,1297,1298,1299,1300,1301,1302,1303,1304,1305,1306,1307,1308,1309,1310,1311,1312,1313,1314,1315,1316,1317,1318,1319,1320,1321,1322,1323,1324,1325,1326,1327,1328,1329,1330,1331,1332,1333,1334,1335,1336,1337,1338,1339,1340,1341,1342,1343,1344,1345,1346,1347,1348,1349,1350,1351,1352,1353,1354,1355,1356,1357,1358,1359,1360,1361,1362,1363,1364,1365,1366,1367,1368,1369,1370,1371,1372,1373,1374,1375,1376,1377,1378,1379,1380,1381,1382,1383,1384,1385,1386,1387,1388,1389,1390,1391,1392,1393,1394,1395,1396,1397,1398,1399,1400,1401,1402,1403,1404,1405,1406,1407,1408,1409,1410,1411,1412,1413,1414,1415,1416,1417,1418,1419,1420,1421,1422,1423,1424,1425,1426,1427,1428,1429,1430,1431,1432,1433,1434,1435,1436,1437,1438,1439,1440])
lfpchannels= np.array([10,30,50,70,90,110,130,150,170,190,210,230,250,270,290,310,330,350,370,390,410,430,450,470,490,510,530,550,570,590,610,630,650,670,690,710,730,750,770,790,810,830,850,870,890,910,930,950,970,990,1010,1030,1050,1070,1090,1110,1130,1150,1170,1190,1210,1230,1250,1270,1290,1310,1330,1350,1370,1390,1410,1430])
refchannels= np.array([57,58,59,60,61,62,63,64,177,178,179,180,181,182,183,184,297,298,299,300,301,302,303,304,417,418,419,420,421,422,423,424,537,538,539,540,541,542,543,544,657,658,659,660,661,662,663,664,777,778,779,780,781,782,783,784,897,898,899,900,901,902,903,904,1017,1018,1019,1020,1021,1022,1023,1024,1137,1138,1139,1140,1141,1142,1143,1144,1257,1258,1259,1260,1261,1262,1263,1264,1377,1378,1379,1380,1381,1382,1383,1384])
#Group 10 out of the
#OUTchannels= np.array([1081,1082,1083,1084,1085,1086,1087,1088,1089,1090,1091,1092,1093,1094,1095,1096,1097,1098,1099,1100,1101,1102,1103,1104,1105,1106,1107,1108,1109,1110,1111,1112,1113,1114,1115,1116,1117,1118,1119,1120,1121,1122,1123,1124,1125,1126,1127,1128,1129,1130,1131,1132,1133,1134,1135,1136,1137,1138,1139,1140,1141,1142,1143,1144,1145,1146,1147,1148,1149,1150,1151,1152,1153,1154,1155,1156,1157,1158,1159,1160,1161,1162,1163,1164,1165,1166,1167,1168,1169,1170,1171,1172,1173,1174,1175,1176,1177,1178,1179,1180,1181,1182,1183,1184,1185,1186,1187,1188,1189,1190,1191,1192,1193,1194,1195,1196,1197,1198,1199,1200])
#Half of Group10 IN
#OUTchannels= np.arange(1150, 1200)
channels_idx_bad = np.concatenate((OFFchannels-1, lfpchannels-1, refchannels-1),axis=0)
#channels_idx_ON = (np.array([x for x in np.arange(0,1440) if x not in channels_idx_bad]))

AP = (np.array([x for x in AP if x not in channels_idx_bad]))

plt.figure()
offset_microvolt= 200
for i in np.arange(0, np.shape(AP)[0]):
    plt.plot(raw_data[AP[i], index1:index2].T + i*offset_microvolt,color='k', linewidth=0.6)
    plt.title(np.str(( AP,index1,index2)))
    plt.show()



OFFchannels= np.array([1201,1202,1203,1204,1205,1206,1207,1208,1209,1210,1211,1212,1213,1214,1215,1216,1217,1218,1219,1220,1221,1222,1223,1224,1225,1226,1227,1228,1229,1230,1231,1232,1233,1234,1235,1236,1237,1238,1239,1240,1241,1242,1243,1244,1245,1246,1247,1248,1249,1250,1251,1252,1253,1254,1255,1256,1257,1258,1259,1260,1261,1262,1263,1264,1265,1266,1267,1268,1269,1270,1271,1272,1273,1274,1275,1276,1277,1278,1279,1280,1281,1282,1283,1284,1285,1286,1287,1288,1289,1290,1291,1292,1293,1294,1295,1296,1297,1298,1299,1300,1301,1302,1303,1304,1305,1306,1307,1308,1309,1310,1311,1312,1313,1314,1315,1316,1317,1318,1319,1320,1321,1322,1323,1324,1325,1326,1327,1328,1329,1330,1331,1332,1333,1334,1335,1336,1337,1338,1339,1340,1341,1342,1343,1344,1345,1346,1347,1348,1349,1350,1351,1352,1353,1354,1355,1356,1357,1358,1359,1360,1361,1362,1363,1364,1365,1366,1367,1368,1369,1370,1371,1372,1373,1374,1375,1376,1377,1378,1379,1380,1381,1382,1383,1384,1385,1386,1387,1388,1389,1390,1391,1392,1393,1394,1395,1396,1397,1398,1399,1400,1401,1402,1403,1404,1405,1406,1407,1408,1409,1410,1411,1412,1413,1414,1415,1416,1417,1418,1419,1420,1421,1422,1423,1424,1425,1426,1427,1428,1429,1430,1431,1432,1433,1434,1435,1436,1437,1438,1439,1440])
lfpchannels= np.array([10,30,50,70,90,110,130,150,170,190,210,230,250,270,290,310,330,350,370,390,410,430,450,470,490,510,530,550,570,590,610,630,650,670,690,710,730,750,770,790,810,830,850,870,890,910,930,950,970,990,1010,1030,1050,1070,1090,1110,1130,1150,1170,1190,1210,1230,1250,1270,1290,1310,1330,1350,1370,1390,1410,1430])
refchannels= np.array([57,58,59,60,61,62,63,64,177,178,179,180,181,182,183,184,297,298,299,300,301,302,303,304,417,418,419,420,421,422,423,424,537,538,539,540,541,542,543,544,657,658,659,660,661,662,663,664,777,778,779,780,781,782,783,784,897,898,899,900,901,902,903,904,1017,1018,1019,1020,1021,1022,1023,1024,1137,1138,1139,1140,1141,1142,1143,1144,1257,1258,1259,1260,1261,1262,1263,1264,1377,1378,1379,1380,1381,1382,1383,1384])
#Group 10 out of the
#OUTchannels= np.array([1081,1082,1083,1084,1085,1086,1087,1088,1089,1090,1091,1092,1093,1094,1095,1096,1097,1098,1099,1100,1101,1102,1103,1104,1105,1106,1107,1108,1109,1110,1111,1112,1113,1114,1115,1116,1117,1118,1119,1120,1121,1122,1123,1124,1125,1126,1127,1128,1129,1130,1131,1132,1133,1134,1135,1136,1137,1138,1139,1140,1141,1142,1143,1144,1145,1146,1147,1148,1149,1150,1151,1152,1153,1154,1155,1156,1157,1158,1159,1160,1161,1162,1163,1164,1165,1166,1167,1168,1169,1170,1171,1172,1173,1174,1175,1176,1177,1178,1179,1180,1181,1182,1183,1184,1185,1186,1187,1188,1189,1190,1191,1192,1193,1194,1195,1196,1197,1198,1199,1200])
#Half of Group10 IN
OUTchannels= np.arange(1150, 1200)
channels_idx_bad = np.concatenate((OFFchannels-1, lfpchannels-1, refchannels-1, OUTchannels),axis=0)
channels_idx_ON = (np.array([x for x in np.arange(0,1440) if x not in channels_idx_bad]))

APThalamus= channels_idx_ON[:106*4]
APhipocampus= channels_idx_ON[106*4: 106*7]
APcortex = channels_idx_ON[106*7: 1012]
AP = np.concatenate((APThalamus,APhipocampus,APcortex), axis=0)

