
from os.path import join
import numpy as np
import catboost as cb
from sklearn.metrics import mean_squared_error
from sklearn.dummy import DummyClassifier


folder = r'/ceph/scratch/gdimitriadis/Neuroseeker/AK_33.1/2018_04_30-11_38/Analysis/NNs/Data/PCA_and_Tsne/data_samplesEvery2Frames_5secslong_tsneOfFrame'
folder = r'/ceph/scratch/gdimitriadis/Neuroseeker/AK_33.1/2018_04_30-11_38/Analysis/NeuropixelSimulations/Sparce/NNs/Data/PCA_And_Tsne/data_samplesEvery2Frames_5secslong_tsneOfFrame'
headers = np.load(join(folder, 'binary_headers.npz'), allow_pickle=True)

dtype = headers['dtype'][0]
shape_X = tuple(headers['shape_X'])
shape_Y = tuple(headers['shape_Y'])


X_buffer = np.memmap(join(folder, 'X_buffer.npy'), dtype=dtype, mode='r', shape=shape_X)
Y_buffer = np.memmap(join(folder, 'Y_buffer.npy'), dtype=dtype, mode='r', shape=shape_Y)


train_ind = np.arange(int(0.7*shape_X[0]))
test_ind = np.arange(int(0.7*shape_X[0]), shape_X[0])


X_test = X_buffer[test_ind,:]
X_train = X_buffer[train_ind,:]
Y_test = Y_buffer[test_ind,:]
Y_train = Y_buffer[train_ind,:]

booster = cb.CatBoost(params={'iterations':10000, 'verbose':10, 'loss_function':'MultiRMSE'})

booster.fit(X_train, Y_train)

test_preds = booster.predict(X_test)
train_preds = booster.predict(X_train)

test_preds = []
mse_test = []
for i in np.arange(0, len(test_ind), int(len(test_ind)/50)):
    test_preds.append(booster.predict(X_test[i:i+50, :]))
    mse_test.append(mean_squared_error(Y_test[i:i+50, :], test_preds[-1]))

mean_squared_error(Y_test, test_preds)
mean_squared_error(Y_train, train_preds)


mse_test_full = [63.10423914472622, 266.7200575497373, 372.81424435461116, 290.2370491038564, 217.39021837171165, 399.5518922162935, 338.4782022313709, 237.41878044803144, 3002.5422332372045, 1352.1334148144551, 88.03336202654235, 593.6737392315463, 532.6244870221343, 297.0921299166548, 962.2006868846173, 507.8664019177454, 874.4152858102638, 164.5957033282251, 194.64343327835843, 161.26401392120283, 1561.7497257889497, 2380.511531316961, 206.74225022809102, 791.7306144443522, 599.8707086600673, 1318.2164218218513, 253.8339059232195, 1599.4311030732629, 5007.407924062956, 54.56036561468082, 20.428429020878195, 69.49725873983553, 60.521997746254236, 29.701818697445358, 1372.1107589393291, 1272.9041807501806, 99.69874904302787, 785.6195135636991, 501.40816503558125, 601.6707946067891, 759.7430324060111, 972.4731301843966, 976.809697291928, 489.44166713582854, 119.96917259990991, 74.60447494905377, 530.0889423517531, 478.6303000496132, 340.33447481967613, 894.0182403271925, 2804.141259855771]
mse_test_sparce = [74.83262656523223, 78.22973283889345, 411.95651264716065, 257.60033710436744, 192.6602951655395, 231.86949409069456, 577.4271231443859, 529.0924604141916, 2797.0237062076253, 1869.5981342837508, 219.7007404831694, 251.95211743970725, 587.3337313565014, 523.6233204090804, 1089.241226100177, 629.1949536871637, 706.6356224646809, 369.6897959453984, 449.3617377170716, 836.3824272370543, 1281.6520229487305, 2758.2770506314173, 106.72056425370607, 1221.8778664266733, 713.2897942029526, 2210.7148389436215, 771.7632568178547, 1499.057336289648, 5937.896222265645, 27.91436049933644, 24.354515469290796, 23.773292283164544, 49.776206745494385, 15.527842111944995, 1584.3276882986297, 3239.9037912721506, 151.19869129422037, 393.7477333352034, 320.13335427458225, 176.77946023784617, 1205.1894606961337, 578.1067578288706, 603.0344099553965, 350.18059830221677, 35.290931852711196, 11.307873123301034, 328.19814379878517, 338.6791865471131, 102.65512252602478, 605.6962186079056, 2791.2815356881547]



dummy = DummyClassifier(random_state=0)
dummy.fit(X_train, Y_train)


test_preds_dummy = dummy.predict(X_test)
train_preds_dummy = dummy.predict(X_train)
test_preds_dummy = []
mse_test_dummy = []
for i in np.arange(0, len(test_ind), int(len(test_ind)/50)):
    test_preds_dummy.append(dummy.predict(X_test[i:i+50, :]))
    mse_test_dummy.append(mean_squared_error(Y_test[i:i+50, :], test_preds_dummy[-1]))
    print(i)

mse_test_dummy = [366.99573, 1295.5002, 1256.0547, 1996.1669, 1518.249, 1274.923, 1571.036, 1363.59, 2527.8972, 1679.5035, 7545.971, 10084.925, 2273.591, 907.77185, 2685.2747, 405.76733, 5239.1553, 1448.3552, 2954.2156, 3208.9734, 3581.5564, 7169.1533, 3763.8076, 2237.7524, 4277.7373, 10977.422, 9168.588, 7185.9854, 1076.3837, 2054.6104, 2466.4116, 2291.8823, 5596.6797, 3582.2236, 4932.3384, 7562.0146, 6133.0283, 2121.0532, 4225.4985, 4167.8135, 3379.8132, 2310.6428, 1612.5183, 1153.9722, 309.65335, 450.34634, 867.98816, 1403.4066, 733.6473, 1452.5902, 3281.8901]
