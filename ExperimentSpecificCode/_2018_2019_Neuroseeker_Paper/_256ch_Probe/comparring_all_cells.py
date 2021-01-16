
from os.path import join
import numpy as np
from sklearn.decomposition import PCA
from spikesorting_tsne import tsne, io_with_cpp as tsne_io

probe_layout_folder = r'E:\Software\Develop\Source\Repos\Python35Projects\TheMeaningOfBrain\Layouts\Probes'

base_save_folder = r'D:\_256channels'
cell_folders = [r'Co2', r'Co3', r'Co4', r'H3', r'St2', r'T1']
decimation_type_folders = [r'Full_Channels', r'Decimated_to_Neuroseeker_Density']

barnes_hut_exe_dir = r'E:\Software\Develop\Source\Repos\spikesorting_tsne_bhpart\Barnes_Hut\win\x64\Release'

# ----------------------------------------------------------------------------------------------------------------------
# <editor-fold desc = "GET THE COUNT OF ALL THE CELLS (BOTH AT FULL DENSITY AND DECIMATED)"

for cell_folder in cell_folders:
    for decimation_type_folder in decimation_type_folders:
        analysis_folder = join(base_save_folder, cell_folder,'Analysis')
        kilosort_folder = join(analysis_folder, decimation_type_folder, 'Kilosort')

        template_marking = np.load(join(kilosort_folder, 'template_marking.npy'))

        print('------- Cell: {}, Type: {} ---------'.format(cell_folder, decimation_type_folder))
        print('Good Cells: {}'.format(len(template_marking) - len(np.argwhere(template_marking == 0))))
        print('Single: {}'.format(len(np.argwhere(template_marking == 1))))
        print('Contaminated: {}'.format(len(np.argwhere(template_marking == 2))))
        print('Putative: {}'.format(len(np.argwhere(template_marking == 3))))
        print('Non Multi: {}'.format(len(np.argwhere(template_marking == 1)) + len(np.argwhere(template_marking == 2))
                                     + len(np.argwhere(template_marking == 3))))
        print('Multi: {}'.format(len(np.argwhere(template_marking == 4))))
        print('-------------------------------------------------')


# </editor-fold>
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# <editor-fold desc = "T-SNE ALL CELLS TOGETHER TO SEE IF THEY CAN BE DIFFERENTIATED"


# Put the template waveforms together
all_cells_full_templates = []
all_cells_deci_templates = []
indices_full = []
indices_deci = []

for cell_folder in cell_folders:
    for decimation_type_folder in decimation_type_folders:
        analysis_folder = join(base_save_folder, cell_folder, 'Analysis')
        kilosort_folder = join(analysis_folder, decimation_type_folder, 'Kilosort')

        data = np.load(join(kilosort_folder, 'avg_spike_template.npy'))

        if decimation_type_folder == decimation_type_folders[0]:
            indices_full.append([len(all_cells_full_templates), len(all_cells_full_templates) + data.shape[0]])
            if len(all_cells_full_templates) == 0:
                all_cells_full_templates = np.copy(data)
            else:
                all_cells_full_templates = np.concatenate((all_cells_full_templates, data), axis=0)
        else:
            indices_deci.append([len(all_cells_deci_templates), len(all_cells_deci_templates) + data.shape[0]])
            if len(all_cells_deci_templates) == 0:
                all_cells_deci_templates = np.copy(data)
            else:
                all_cells_deci_templates = np.concatenate((all_cells_deci_templates, data), axis=0)


# T-sne the full

tsne_folder_full = join(base_save_folder, 'common_results', 'tsne', 'PCs_of_full_waveforms')

all_cells_full_templates_flat = all_cells_full_templates.reshape((all_cells_full_templates.shape[0],
                                                                  all_cells_full_templates.shape[1] * all_cells_full_templates.shape[2]))

pca_sr_full = PCA()
pcs_ar_full = pca_sr_full.fit_transform(all_cells_full_templates_flat)

number_of_top_pcs = 40
num_dims = 2
perplexity = 30
theta = 0.3
eta = 200
exageration = 12
iterations = 4000
random_seed = 1
verbose = 2
tsne_result_full = tsne.t_sne(pcs_ar_full[:, :number_of_top_pcs], tsne_folder_full, barnes_hut_exe_dir,
                                         num_dims=num_dims,
                                         perplexity=perplexity, theta=theta, eta=eta, exageration=exageration,
                                         iterations=iterations, random_seed=random_seed, verbose=verbose)

tsne_result_full = tsne_io.load_tsne_result(tsne_folder_full)

plt.figure(1)
plt.scatter(tsne_result_full[:, 0], tsne_result_full[:, 1], s=3, c='k')

plt.figure(2)
colors = ['k', 'r', 'b', 'y', 'g', 'm']
for ind, i in zip(indices_full, np.arange(len(indices_full))):
    plt.scatter(tsne_result_full[ind[0]:ind[1], 0], tsne_result_full[ind[0]:ind[1], 1], s=10, c=colors[i])
plt.legend(cell_folders)


# T-sne the decimated

tsne_folder_deci = join(base_save_folder, 'common_results', 'tsne', 'PCs_of_decimated_waveforms')

all_cells_deci_templates_flat = all_cells_deci_templates.reshape((all_cells_deci_templates.shape[0],
                                                                  all_cells_deci_templates.shape[1] * all_cells_deci_templates.shape[2]))

pca_sr_deci = PCA()
pcs_ar_deci = pca_sr_deci.fit_transform(all_cells_deci_templates_flat)

number_of_top_pcs = 40
num_dims = 2
perplexity = 30
theta = 0.3
eta = 200
exageration = 12
iterations = 4000
random_seed = 1
verbose = 2
tsne_result_full = tsne.t_sne(pcs_ar_deci[:, :number_of_top_pcs], tsne_folder_deci, barnes_hut_exe_dir,
                                         num_dims=num_dims,
                                         perplexity=perplexity, theta=theta, eta=eta, exageration=exageration,
                                         iterations=iterations, random_seed=random_seed, verbose=verbose)

tsne_result_deci = tsne_io.load_tsne_result(tsne_folder_deci)

plt.figure(3)
plt.scatter(tsne_result_deci[:, 0], tsne_result_deci[:, 1], s=3, c='k')

plt.figure(4)
colors = ['k', 'r', 'b', 'y', 'g', 'm']
for ind, i in zip(indices_deci, np.arange(len(indices_deci))):
    plt.scatter(tsne_result_deci[ind[0]:ind[1], 0], tsne_result_deci[ind[0]:ind[1], 1], s=10, c=colors[i])
plt.legend(cell_folders)


# </editor-fold>
# ----------------------------------------------------------------------------------------------------------------------
