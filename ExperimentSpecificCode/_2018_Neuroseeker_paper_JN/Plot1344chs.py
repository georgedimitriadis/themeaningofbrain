import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import scipy.stats as stats
import IO.ephys as ephys


#File names-------------------------------------------------------------------------------------------------------------
#bin files are filtered with a band-pass of 500-3,500 Hz and
#when using the external reference the median signal within each group across the recording
# electrodes from the respective group was subtracted

#17h58, 10 active groups, ref internal
time = '17_58_26.bin'
binary_data_filename = r'Z:\labs2\kampff\n\Neuroseeker Probe Recordings\Neuroseeker_2017_08_08\Data'
binary_data_filename= os.path.join(binary_data_filename, time)

#18h26, 10 active groups, ref external
binary_data_filename =r'Z:\labs2\kampff\n\Neuroseeker Probe Recordings\Neuroseeker_2017_08_08\Data\18_26_30.bin'

#18h26, 10 active groups, ref external, LFPS after REF each group
time = r'18_26_30_afterREF_LFPs.bin'
binary_data_filename = r'Z:\labs2\kampff\n\Neuroseeker Probe Recordings\Neuroseeker_2017_08_08\Data'
binary_data_filename= os.path.join(binary_data_filename, time)


#Open Data--------------------------------------------------------------------------------------------------------------
number_of_channels_in_binary_file = 1440
sampling_frequency = 20000
raw_data = np.memmap(binary_data_filename, dtype=np.int16, mode='r')
number_of_timepoints_in_raw = int(raw_data.shape[0] / number_of_channels_in_binary_file)
raw_data = np.reshape(raw_data, (number_of_channels_in_binary_file, number_of_timepoints_in_raw), order='F')


#Plot some APs traces from different regions----------------------------------------------------------------------------
#figure 2A
#18h26, 10 active groups, ref external
#10 regions ON and half of the group 10  is outside of the brain

#Electrode numbers for each region
#cortex regions
APC0=np.arange(849,857)
APC1=np.arange(985,1001)
APC2=np.arange(1081,1097)
APC4= np.arange(1145,1153)
#hipocampus regions
APH0=np.arange(473,481)
APH1=np.arange(577, 585)
APH2 =np.arange(697,705)
APH3=np.arange(769,777)
#thalamus regions
APT1= np.arange(17,25)
APT2= np.arange(137, 145)
APT3= np.arange(313,321)
APT4=np.arange(369,377)
#AP= np.concatenate((APC0,APC1,APC2,APC3,APC4),axis=0)
AP= np.concatenate((APT1,APT2,APT3,APT4,APH0,APH1,APH2,APH3,APC0,APC1,APC2,APC4),axis=0)
AP =AP-1


#5 seconds traces
time_samples = 100000
index1 = np.int(raw_data.shape[1]/10*6) - 2000
index2 = np.int(index1 + time_samples)
#time_samples = 200000.0
#index1 = np.int(raw_data.shape[1]/10*9) - 2000
#index2 = np.int(index1 + time_samples)


#Final electrode numbers wout lfps, refs and off channels, 106 AP traces
OFFchannels= np.array([1201,1202,1203,1204,1205,1206,1207,1208,1209,1210,1211,1212,1213,1214,1215,1216,1217,1218,1219,1220,1221,1222,1223,1224,1225,1226,1227,1228,1229,1230,1231,1232,1233,1234,1235,1236,1237,1238,1239,1240,1241,1242,1243,1244,1245,1246,1247,1248,1249,1250,1251,1252,1253,1254,1255,1256,1257,1258,1259,1260,1261,1262,1263,1264,1265,1266,1267,1268,1269,1270,1271,1272,1273,1274,1275,1276,1277,1278,1279,1280,1281,1282,1283,1284,1285,1286,1287,1288,1289,1290,1291,1292,1293,1294,1295,1296,1297,1298,1299,1300,1301,1302,1303,1304,1305,1306,1307,1308,1309,1310,1311,1312,1313,1314,1315,1316,1317,1318,1319,1320,1321,1322,1323,1324,1325,1326,1327,1328,1329,1330,1331,1332,1333,1334,1335,1336,1337,1338,1339,1340,1341,1342,1343,1344,1345,1346,1347,1348,1349,1350,1351,1352,1353,1354,1355,1356,1357,1358,1359,1360,1361,1362,1363,1364,1365,1366,1367,1368,1369,1370,1371,1372,1373,1374,1375,1376,1377,1378,1379,1380,1381,1382,1383,1384,1385,1386,1387,1388,1389,1390,1391,1392,1393,1394,1395,1396,1397,1398,1399,1400,1401,1402,1403,1404,1405,1406,1407,1408,1409,1410,1411,1412,1413,1414,1415,1416,1417,1418,1419,1420,1421,1422,1423,1424,1425,1426,1427,1428,1429,1430,1431,1432,1433,1434,1435,1436,1437,1438,1439,1440])
lfpchannels= np.array([10,30,50,70,90,110,130,150,170,190,210,230,250,270,290,310,330,350,370,390,410,430,450,470,490,510,530,550,570,590,610,630,650,670,690,710,730,750,770,790,810,830,850,870,890,910,930,950,970,990,1010,1030,1050,1070,1090,1110,1130,1150,1170,1190,1210,1230,1250,1270,1290,1310,1330,1350,1370,1390,1410,1430])
refchannels= np.array([57,58,59,60,61,62,63,64,177,178,179,180,181,182,183,184,297,298,299,300,301,302,303,304,417,418,419,420,421,422,423,424,537,538,539,540,541,542,543,544,657,658,659,660,661,662,663,664,777,778,779,780,781,782,783,784,897,898,899,900,901,902,903,904,1017,1018,1019,1020,1021,1022,1023,1024,1137,1138,1139,1140,1141,1142,1143,1144,1257,1258,1259,1260,1261,1262,1263,1264,1377,1378,1379,1380,1381,1382,1383,1384])
#Group 10 OUT
#OUTchannels= np.array([1081,1082,1083,1084,1085,1086,1087,1088,1089,1090,1091,1092,1093,1094,1095,1096,1097,1098,1099,1100,1101,1102,1103,1104,1105,1106,1107,1108,1109,1110,1111,1112,1113,1114,1115,1116,1117,1118,1119,1120,1121,1122,1123,1124,1125,1126,1127,1128,1129,1130,1131,1132,1133,1134,1135,1136,1137,1138,1139,1140,1141,1142,1143,1144,1145,1146,1147,1148,1149,1150,1151,1152,1153,1154,1155,1156,1157,1158,1159,1160,1161,1162,1163,1164,1165,1166,1167,1168,1169,1170,1171,1172,1173,1174,1175,1176,1177,1178,1179,1180,1181,1182,1183,1184,1185,1186,1187,1188,1189,1190,1191,1192,1193,1194,1195,1196,1197,1198,1199,1200])
#Half of Group10 OUT
#OUTchannels= np.arange(1150, 1200)
channels_idx_bad = np.concatenate((OFFchannels-1, lfpchannels-1, refchannels-1),axis=0)
#channels_idx_ON = (np.array([x for x in np.arange(0,1440) if x not in channels_idx_bad]))
AP = (np.array([x for x in AP if x not in channels_idx_bad]))


#plot
plt.figure()
offset_microvolt= 200
for i in np.arange(0, np.shape(AP)[0]):
    plt.plot(raw_data[AP[i], index1:index2].T + i*offset_microvolt,color='b', linewidth=0.8)
    plt.title(np.str(( AP,index1,index2)))
    plt.show()



#LFPs from different regions---------------------------------------------------------------------------------------------
#figure 2A
#electrode numbers, 60 LFP traces
APlfp = np.arange(9, 1430, 20)
OFFchannels= np.array([1201,1202,1203,1204,1205,1206,1207,1208,1209,1210,1211,1212,1213,1214,1215,1216,1217,1218,1219,1220,1221,1222,1223,1224,1225,1226,1227,1228,1229,1230,1231,1232,1233,1234,1235,1236,1237,1238,1239,1240,1241,1242,1243,1244,1245,1246,1247,1248,1249,1250,1251,1252,1253,1254,1255,1256,1257,1258,1259,1260,1261,1262,1263,1264,1265,1266,1267,1268,1269,1270,1271,1272,1273,1274,1275,1276,1277,1278,1279,1280,1281,1282,1283,1284,1285,1286,1287,1288,1289,1290,1291,1292,1293,1294,1295,1296,1297,1298,1299,1300,1301,1302,1303,1304,1305,1306,1307,1308,1309,1310,1311,1312,1313,1314,1315,1316,1317,1318,1319,1320,1321,1322,1323,1324,1325,1326,1327,1328,1329,1330,1331,1332,1333,1334,1335,1336,1337,1338,1339,1340,1341,1342,1343,1344,1345,1346,1347,1348,1349,1350,1351,1352,1353,1354,1355,1356,1357,1358,1359,1360,1361,1362,1363,1364,1365,1366,1367,1368,1369,1370,1371,1372,1373,1374,1375,1376,1377,1378,1379,1380,1381,1382,1383,1384,1385,1386,1387,1388,1389,1390,1391,1392,1393,1394,1395,1396,1397,1398,1399,1400,1401,1402,1403,1404,1405,1406,1407,1408,1409,1410,1411,1412,1413,1414,1415,1416,1417,1418,1419,1420,1421,1422,1423,1424,1425,1426,1427,1428,1429,1430,1431,1432,1433,1434,1435,1436,1437,1438,1439,1440])
refchannels= np.array([57,58,59,60,61,62,63,64,177,178,179,180,181,182,183,184,297,298,299,300,301,302,303,304,417,418,419,420,421,422,423,424,537,538,539,540,541,542,543,544,657,658,659,660,661,662,663,664,777,778,779,780,781,782,783,784,897,898,899,900,901,902,903,904,1017,1018,1019,1020,1021,1022,1023,1024,1137,1138,1139,1140,1141,1142,1143,1144,1257,1258,1259,1260,1261,1262,1263,1264,1377,1378,1379,1380,1381,1382,1383,1384])
channels_idx_bad_lfp = np.concatenate((OFFchannels-1, refchannels-1),axis=0)
#channels_idx_bad_lfp = refchannels-1
#channels_idx_ON = (np.array([x for x in np.arange(0,1440) if x not in channels_idx_bad]))
APlfp = (np.array([x for x in APlfp if x not in channels_idx_bad_lfp]))
#APlfp = [109,269,409,469,529,649,689,789,849,989]


#5 seconds traces
time_samples = 100000
#time_samples = 20000
index1 = np.int(raw_data.shape[1]/10*6) - 2000
index2 = np.int(index1 + time_samples)
#index1 = np.int(raw_data.shape[1]/10*6) - 2000 + 35000
#index2 = np.int(index1 + time_samples)


#plot
#offset_microvolt = 1000
offset_microvolt = 700
plt.figure()
for i in np.arange(0, np.shape(APlfp)[0]):
    plt.plot(raw_data[APlfp[i], index1:index2].T + i*offset_microvolt, linewidth=0.8)
   # plt.plot(raw_data[APlfp[i], index1:index2].T + i*offset_microvolt, color='b', linewidth=0.8)
    plt.title(np.str((APlfp,index1,index2)))
    plt.show()



#Plot all traces from each group----------------------------------------------------------------------------------------
groups= np.arange(1,10)

#5 seconds traces
time_samples = 100000.0
index1 = np.int(raw_data.shape[1]/10*6)
index2 = np.int(index1 + time_samples)

#plot
plt.figure()
offset_microvolt= 200
fig_folder=r'Z:\n\Neuroseeker Probe Recordings\Neuroseeker_2017_08_08\Analysis'
for g in groups:
    plt.figure()
    AP = np.arange((g-1)*120, g*120)
    for i in np.arange(0, np.shape(AP)[0]):
        plt.plot(raw_data[AP[i], index1:index2].T + i * offset_microvolt)
        plt.title(np.str((AP, index1, index2)))
        plt.show()
        #fig_filename = os.path.join(fig_folder + '\\'+ str(g) + 'group.png')
        #plt.savefig(fig_filename)



#Plot all APs traces from 10 groups-------------------------------------------------------------------------------------
#Figure 1 Supplementary Material

#electrode numbers wout lfps, refs and off channels, 1060 AP traces
AP= np.arange(0, 1200)
OFFchannels= np.array([1201,1202,1203,1204,1205,1206,1207,1208,1209,1210,1211,1212,1213,1214,1215,1216,1217,1218,1219,1220,1221,1222,1223,1224,1225,1226,1227,1228,1229,1230,1231,1232,1233,1234,1235,1236,1237,1238,1239,1240,1241,1242,1243,1244,1245,1246,1247,1248,1249,1250,1251,1252,1253,1254,1255,1256,1257,1258,1259,1260,1261,1262,1263,1264,1265,1266,1267,1268,1269,1270,1271,1272,1273,1274,1275,1276,1277,1278,1279,1280,1281,1282,1283,1284,1285,1286,1287,1288,1289,1290,1291,1292,1293,1294,1295,1296,1297,1298,1299,1300,1301,1302,1303,1304,1305,1306,1307,1308,1309,1310,1311,1312,1313,1314,1315,1316,1317,1318,1319,1320,1321,1322,1323,1324,1325,1326,1327,1328,1329,1330,1331,1332,1333,1334,1335,1336,1337,1338,1339,1340,1341,1342,1343,1344,1345,1346,1347,1348,1349,1350,1351,1352,1353,1354,1355,1356,1357,1358,1359,1360,1361,1362,1363,1364,1365,1366,1367,1368,1369,1370,1371,1372,1373,1374,1375,1376,1377,1378,1379,1380,1381,1382,1383,1384,1385,1386,1387,1388,1389,1390,1391,1392,1393,1394,1395,1396,1397,1398,1399,1400,1401,1402,1403,1404,1405,1406,1407,1408,1409,1410,1411,1412,1413,1414,1415,1416,1417,1418,1419,1420,1421,1422,1423,1424,1425,1426,1427,1428,1429,1430,1431,1432,1433,1434,1435,1436,1437,1438,1439,1440])
lfpchannels= np.array([10,30,50,70,90,110,130,150,170,190,210,230,250,270,290,310,330,350,370,390,410,430,450,470,490,510,530,550,570,590,610,630,650,670,690,710,730,750,770,790,810,830,850,870,890,910,930,950,970,990,1010,1030,1050,1070,1090,1110,1130,1150,1170,1190,1210,1230,1250,1270,1290,1310,1330,1350,1370,1390,1410,1430])
refchannels= np.array([57,58,59,60,61,62,63,64,177,178,179,180,181,182,183,184,297,298,299,300,301,302,303,304,417,418,419,420,421,422,423,424,537,538,539,540,541,542,543,544,657,658,659,660,661,662,663,664,777,778,779,780,781,782,783,784,897,898,899,900,901,902,903,904,1017,1018,1019,1020,1021,1022,1023,1024,1137,1138,1139,1140,1141,1142,1143,1144,1257,1258,1259,1260,1261,1262,1263,1264,1377,1378,1379,1380,1381,1382,1383,1384])
#Group 10 OUT
#OUTchannels= np.array([1081,1082,1083,1084,1085,1086,1087,1088,1089,1090,1091,1092,1093,1094,1095,1096,1097,1098,1099,1100,1101,1102,1103,1104,1105,1106,1107,1108,1109,1110,1111,1112,1113,1114,1115,1116,1117,1118,1119,1120,1121,1122,1123,1124,1125,1126,1127,1128,1129,1130,1131,1132,1133,1134,1135,1136,1137,1138,1139,1140,1141,1142,1143,1144,1145,1146,1147,1148,1149,1150,1151,1152,1153,1154,1155,1156,1157,1158,1159,1160,1161,1162,1163,1164,1165,1166,1167,1168,1169,1170,1171,1172,1173,1174,1175,1176,1177,1178,1179,1180,1181,1182,1183,1184,1185,1186,1187,1188,1189,1190,1191,1192,1193,1194,1195,1196,1197,1198,1199,1200])
#Half of Group10 OUT
#OUTchannels= np.arange(1150, 1200)
channels_idx_bad = np.concatenate((OFFchannels-1, lfpchannels-1, refchannels-1),axis=0)
AP = (np.array([x for x in AP if x not in channels_idx_bad]))


#500 ms traces
time_samples = 10000
index1 = np.int(raw_data.shape[1]/10*6) - 2000
index2 = np.int(index1 + time_samples)


# plot
plt.figure()
offset_microvolt = 200
for i in np.arange(0, np.shape(AP)[0]):
    plt.plot(raw_data[AP[i], index1:index2].T + i * offset_microvolt, color='b', linewidth=0.8)
    plt.title(np.str((AP, index1, index2)))
    plt.show()


#Plot all traces from each region with individual colors----------------------------------------------------------------

#500 ms traces
time_samples = 10000
index1 = np.int(raw_data.shape[1]/10*6) - 2000
index2 = np.int(index1 + time_samples)

#plot
plt.figure()
offset_microvolt= 200
fig_folder=r'Z:\n\Neuroseeker Probe Recordings\Neuroseeker_2017_08_08\Analysis'

#color for each region
royal_blue = '#4169e1' #group 8-10 cortex
violet = '#d02090' #group 5-7 hipocampus
red = '#ff0000' #group 1-4 thalamus
#lightsteelblue= '#b0c4de'
#navy = '#000080'
#steelblue = '#4682b4'
#blue = '#0000ff'

#Plot example: hipocampus
groups= np.arange(5,8)
for g in groups:
    AP_g = np.arange((g-1)*106, g*106)
    plt.figure()
    for i in AP_g:
        plt.plot(raw_data[AP[i], index1:index2].T + i * offset_microvolt, color = violet, linewidth =1)
        plt.title(np.str((g, index1, index2)))
        #plt.delaxes()
        plt.show()

