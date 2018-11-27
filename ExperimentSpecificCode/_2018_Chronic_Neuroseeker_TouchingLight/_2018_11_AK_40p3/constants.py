
import os.path as path


probe_layout_folder = r'E:\Software\Develop\Source\Repos\Python35Projects\TheMeaningOfBrain\Layouts\Probes\Neuroseeker'
prb_file = path.join(probe_layout_folder, 'ap_only_prb.txt')


base_save_folder = r'D:\Data\George\Neuroseeker chronic'


experiment_folders = {1: r'AK_40.3_AK_40.4\2018_11_22-10_31'}


NUMBER_OF_CHANNELS_IN_BINARY_FILE = 1368
CAMERA_TTL_PULSES_TIMEPOINT_PERIOD = 158


PROBE_DIMENSIONS = [100, 8100]
POSITION_MULT = 2.25

BRAIN_REGIONS = {'Cortex LPA': 8100, 'Corpus Calosum': 6240, 'CA1': 5780, 'CA2': 5190, 'CA3': 4870,
                 'Thalamus LPLR': 4250, 'Thalamus Po': 3300, 'Thalamus VPM': 2660,
                 'Zona Incerta': 1180, 'Subthalamic Nuclei': 200}
