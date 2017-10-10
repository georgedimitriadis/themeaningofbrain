
from ExperimentSpecificCode.Neuroseeker_Auditory_Double_2017_03_28.Stimulus import arnes_basic_analysis as ba

data_path = r'F:\Neuroseeker\Neuroseeker_2017_03_28_Anesthesia_Auditory_DoubleProbes'


overwrite = True
binwidth = 0.05
stimulus = 'ToneSequence'
pre = 3
post = 3
plot_type = 'psth'
test_pre = 0.4
test_post = 0.1
min_rate = 0.5

ba.tuning(data_path=data_path, overwrite=overwrite, binwidth=binwidth, stimulus=stimulus, frequencies_index=[0, 1, 2, 3],
          pre=pre, post=post, plot_type=plot_type, test_pre=test_pre,  test_post=test_post, min_rate=min_rate)

overwrite = True
binwidth = 0.05
pre = 2
post = 2
test_pre = 0.5
test_post = 0.2
min_rate = 2
ba.tones(data_path=data_path, overwrite=overwrite, binwidth=binwidth, pre=pre, post=post,
          test_pre=test_pre, test_post=test_post, min_rate=min_rate)



template = 120
df = pd.DataFrame(index=[template])

try:
    temp = df['a'].loc[template]
    temp[7, :] = np.ones(100) *5
except:
    temp = np.empty((10, 100))
    temp[0, :] = np.ones(100) * 3
finally:
    df['a'] = pd.Series([temp], index=df.index)