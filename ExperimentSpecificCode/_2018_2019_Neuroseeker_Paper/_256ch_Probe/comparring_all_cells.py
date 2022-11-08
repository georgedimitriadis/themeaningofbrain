
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from spikesorting_tsne import tsne, io_with_cpp as tsne_io
import pandas as pd
from BrainDataAnalysis.tsne_analysis_functions import fit_dbscan

probe_layout_folder = r'E:\Code\Mine\themeaningofbrain\Layouts\Probes'

base_save_folder = r'D:\_256channels'
cell_folders = [r'Co2', r'Co3', r'Co4', r'CoP1', r'CR1', r'H3', r'St2', r'T1']
decimation_type_folders = [r'Full_Channels', r'Decimated_to_Neuroseeker_Density', r'Decimated_to_Sparce_Density']

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
cells_328_full_templates = []
all_cells_deci_ns_templates = []
cells_328_deci_ns_templates = []
all_cells_deci_sp_templates = []
indices_full = []
indices_full_328 = []
indices_deci_ns = []
indices_deci_ns_328 = []
indices_deci_sp = []

for cell_folder in cell_folders:
    ratio_328_873 = (328 / 873)
    ratio_328_663 = (328 / 663)
    for decimation_type_folder in decimation_type_folders:
        analysis_folder = join(base_save_folder, cell_folder, 'Analysis')
        kilosort_folder = join(analysis_folder, decimation_type_folder, 'Kilosort')

        data = np.load(join(kilosort_folder, 'avg_spike_template.npy'))
        template_marking = np.load(join(kilosort_folder, 'template_marking.npy'))
        template_indices = np.squeeze(np.argwhere(template_marking != 0))

        if decimation_type_folder == decimation_type_folders[0]:
            indices_full.append([len(all_cells_full_templates), len(all_cells_full_templates) + len(template_indices)])
            indices_full_328.append([len(cells_328_full_templates), len(cells_328_full_templates) +
                                     int(ratio_328_873 * len(template_indices))])
            if len(all_cells_full_templates) == 0:
                all_cells_full_templates = np.copy(data[template_indices])
                cells_328_full_templates = np.copy(data[np.random.choice(template_indices, int(ratio_328_873 * len(template_indices)),
                                                                         False)])
            else:
                all_cells_full_templates = np.concatenate((all_cells_full_templates, data[template_indices]), axis=0)
                cells_328_full_templates = np.concatenate((cells_328_full_templates, data[np.random.choice(template_indices,
                                                                                                           int(ratio_328_873 * len(template_indices)),
                                                                                                           False), :, :]),
                                                          axis=0)
        elif decimation_type_folder == decimation_type_folders[1]:
            indices_deci_ns.append([len(all_cells_deci_ns_templates), len(all_cells_deci_ns_templates) + len(template_indices)])
            indices_deci_ns_328.append([len(cells_328_deci_ns_templates), len(cells_328_deci_ns_templates) +
                                     int(ratio_328_663 * len(template_indices))])
            if len(all_cells_deci_ns_templates) == 0:
                all_cells_deci_ns_templates = np.copy(data[template_indices])
                cells_328_deci_ns_templates = np.copy(data[np.random.choice(template_indices, int(ratio_328_663 * len(template_indices)),
                                                                         False)])
            else:
                all_cells_deci_ns_templates = np.concatenate((all_cells_deci_ns_templates, data[template_indices]), axis=0)
                cells_328_deci_ns_templates = np.concatenate((cells_328_deci_ns_templates, data[np.random.choice(template_indices,
                                                                                                           int(ratio_328_663 * len(template_indices)),
                                                                                                           False), : , :]),
                                                             axis=0)
        elif decimation_type_folder == decimation_type_folders[2]:
            indices_deci_sp.append([len(all_cells_deci_sp_templates), len(all_cells_deci_sp_templates) + len(template_indices)])
            if len(all_cells_deci_sp_templates) == 0:
                all_cells_deci_sp_templates = np.copy(data[template_indices])
            else:
                all_cells_deci_sp_templates = np.concatenate((all_cells_deci_sp_templates, data[template_indices]), axis=0)

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
iterations = 10000
random_seed = 0
verbose = 2
tsne_result_full = tsne.t_sne(pcs_ar_full[:, :number_of_top_pcs], tsne_folder_full, barnes_hut_exe_dir,
                                         num_dims=num_dims,
                                         perplexity=perplexity, theta=theta, eta=eta, exageration=exageration,
                                         iterations=iterations, random_seed=random_seed, verbose=verbose)

tsne_result_full = tsne_io.load_tsne_result(tsne_folder_full)

# T-sne the full with 328 randomly picked cells

tsne_folder_full_328 = join(base_save_folder, 'common_results', 'tsne', 'PCs_of_full_waveforms_328_picked_cells')

all_cells_full_templates_flat_328 = cells_328_full_templates.reshape((cells_328_full_templates.shape[0],
                                                                  cells_328_full_templates.shape[1] * cells_328_full_templates.shape[2]))

pca_sr_full_328 = PCA()
pcs_ar_full_328 = pca_sr_full_328.fit_transform(all_cells_full_templates_flat_328)

number_of_top_pcs = 40
num_dims = 2
perplexity = 30
theta = 0.3
eta = 200
exageration = 12
iterations = 10000
random_seed = 0
verbose = 2
tsne_result_full_328 = tsne.t_sne(pcs_ar_full_328[:, :number_of_top_pcs], tsne_folder_full_328, barnes_hut_exe_dir,
                                         num_dims=num_dims,
                                         perplexity=perplexity, theta=theta, eta=eta, exageration=exageration,
                                         iterations=iterations, random_seed=random_seed, verbose=verbose)

tsne_result_full_328 = tsne_io.load_tsne_result(tsne_folder_full_328)

# T-sne the NS decimated

tsne_folder_deci = join(base_save_folder, 'common_results', 'tsne', 'PCs_of_decimated_ns_waveforms')

all_cells_deci_templates_flat = all_cells_deci_ns_templates.reshape((all_cells_deci_ns_templates.shape[0],
                                                                     all_cells_deci_ns_templates.shape[1] * all_cells_deci_ns_templates.shape[2]))

pca_sr_deci = PCA()
pcs_ar_deci = pca_sr_deci.fit_transform(all_cells_deci_templates_flat)

number_of_top_pcs = 40
num_dims = 2
perplexity = 30
theta = 0.3
eta = 200
exageration = 12
iterations = 10000
random_seed = 1
verbose = 2
tsne_result_deci = tsne.t_sne(pcs_ar_deci[:, :number_of_top_pcs], tsne_folder_deci, barnes_hut_exe_dir,
                                         num_dims=num_dims,
                                         perplexity=perplexity, theta=theta, eta=eta, exageration=exageration,
                                         iterations=iterations, random_seed=random_seed, verbose=verbose)

tsne_result_deci = tsne_io.load_tsne_result(tsne_folder_deci)


# T-sne the NS decimated with 328 randomly picked cells

tsne_folder_deci_328 = join(base_save_folder, 'common_results', 'tsne', 'PCs_of_decimated_ns_waveforms_328_picked_cells')

all_cells_deci_templates_flat_328 = cells_328_deci_ns_templates.reshape((cells_328_deci_ns_templates.shape[0],
                                                                  cells_328_deci_ns_templates.shape[1] * cells_328_deci_ns_templates.shape[2]))

pca_sr_deci_328 = PCA()
pcs_ar_deci_328 = pca_sr_deci_328.fit_transform(all_cells_deci_templates_flat_328)

number_of_top_pcs = 40
num_dims = 2
perplexity = 30
theta = 0.3
eta = 200
exageration = 12
iterations = 10000
random_seed = 0
verbose = 2
tsne_result_deci_328 = tsne.t_sne(pcs_ar_deci_328[:, :number_of_top_pcs], tsne_folder_deci_328, barnes_hut_exe_dir,
                                         num_dims=num_dims,
                                         perplexity=perplexity, theta=theta, eta=eta, exageration=exageration,
                                         iterations=iterations, random_seed=random_seed, verbose=verbose)

tsne_result_deci_328 = tsne_io.load_tsne_result(tsne_folder_deci_328)

# T-sne the Sparce decimated

tsne_folder_deci_sparce = join(base_save_folder, 'common_results', 'tsne', 'PCs_of_decimated_sparce_waveforms')

all_cells_deci_sparce_templates_flat = all_cells_deci_sp_templates.reshape((all_cells_deci_sp_templates.shape[0],
                                                                     all_cells_deci_sp_templates.shape[1] * all_cells_deci_sp_templates.shape[2]))

pca_sr_deci_sparce = PCA()
pcs_ar_deci_sparce = pca_sr_deci_sparce.fit_transform(all_cells_deci_sparce_templates_flat)

number_of_top_pcs = 40
num_dims = 2
perplexity = 30
theta = 0.3
eta = 200
exageration = 12
iterations = 10000
random_seed = 1
verbose = 2
tsne_result_deci_sparce = tsne.t_sne(pcs_ar_deci_sparce[:, :number_of_top_pcs], tsne_folder_deci_sparce, barnes_hut_exe_dir,
                                         num_dims=num_dims,
                                         perplexity=perplexity, theta=theta, eta=eta, exageration=exageration,
                                         iterations=iterations, random_seed=random_seed, verbose=verbose)

tsne_result_deci_sparce = tsne_io.load_tsne_result(tsne_folder_deci_sparce)


colors = [(1, 0, 0 ), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (0, 0, 0), (0.2, 0.8, 0.5)]

plt.figure(1)
#colors = ['k', 'r', 'b', 'y', 'g', 'm']
for ind, i in zip(indices_full, np.arange(len(indices_full))):
    plt.scatter(tsne_result_full[ind[0]:ind[1], 0], tsne_result_full[ind[0]:ind[1], 1], s=10, c=colors[i])
plt.legend(cell_folders)

plt.figure(2)
#colors = ['k', 'r', 'b', 'y', 'g', 'm']
for ind, i in zip(indices_full_328, np.arange(len(indices_full_328))):
    plt.scatter(tsne_result_full_328[ind[0]:ind[1], 0], tsne_result_full_328[ind[0]:ind[1], 1], s=10, c=colors[i])
plt.legend(cell_folders)


plt.figure(3)
#colors = ['k', 'r', 'b', 'y', 'g', 'm']
for ind, i in zip(indices_deci_ns, np.arange(len(indices_deci_ns))):
    plt.scatter(tsne_result_deci[ind[0]:ind[1], 0], tsne_result_deci[ind[0]:ind[1], 1], s=10, c=colors[i])
plt.legend(cell_folders)

plt.figure(4)
#colors = ['k', 'r', 'b', 'y', 'g', 'm']
for ind, i in zip(indices_deci_ns_328, np.arange(len(indices_deci_ns_328))):
    plt.scatter(tsne_result_deci_328[ind[0]:ind[1], 0], tsne_result_deci_328[ind[0]:ind[1], 1], s=10, c=colors[i])
plt.legend(cell_folders)


plt.figure(5)
#colors = ['k', 'r', 'b', 'y', 'g', 'm']
for ind, i in zip(indices_deci_sp, np.arange(len(indices_deci_sp))):
    plt.scatter(tsne_result_deci_sparce[ind[0]:ind[1], 0], tsne_result_deci_sparce[ind[0]:ind[1], 1], s=20, c=colors[i])
plt.legend(cell_folders)

eps = 0.035
min_samples = 4

eps = 0.034
min_samples = 4
#db, n_clusters_, labels, core_samples_mask, score = fit_dbscan(tsne_result_full.T, eps, min_samples)
db, n_clusters_, labels, core_samples_mask, score = fit_dbscan(tsne_result_full_328.T, eps, min_samples)
#db, n_clusters_, labels, core_samples_mask, score = fit_dbscan(tsne_result_deci.T, eps, min_samples)
db, n_clusters_, labels, core_samples_mask, score = fit_dbscan(tsne_result_deci_328.T, eps, min_samples)
#db, n_clusters_, labels, core_samples_mask, score = fit_dbscan(tsne_result_deci_sparce.T, eps, min_samples)


clusters_over_eps_and_min_samples = np.empty((10,6,3))
for e, eps in enumerate(np.arange(0.02, 0.06, 0.004)):
    for s, min_samples in enumerate(np.arange(2, 8, 1)):
        db, n_clusters_, labels, core_samples_mask, score = fit_dbscan(tsne_result_full.T, eps, min_samples, show=False)
        clusters_over_eps_and_min_samples[e, s, 0] = n_clusters_
        db, n_clusters_, labels, core_samples_mask, score = fit_dbscan(tsne_result_deci.T, eps, min_samples, show=False)
        clusters_over_eps_and_min_samples[e, s, 1] = n_clusters_
        db, n_clusters_, labels, core_samples_mask, score = fit_dbscan(tsne_result_deci_sparce.T, eps, min_samples, show=False)
        clusters_over_eps_and_min_samples[e, s, 2] = n_clusters_

clusters_over_eps_and_min_samples_percent = np.empty((10,6,2))
clusters_over_eps_and_min_samples_percent[:,:, 0] = clusters_over_eps_and_min_samples[:,:,1] / clusters_over_eps_and_min_samples[:,:,0]
clusters_over_eps_and_min_samples_percent[:,:, 1] = clusters_over_eps_and_min_samples[:,:,2] / clusters_over_eps_and_min_samples[:,:,0]

plt.figure(0)
plt.scatter(X[:, 0], X[:, 1], s=3, c='k')

plt.figure(1)
plt.scatter(tsne_result_full[:, 0], tsne_result_full[:, 1], s=3, c='k')
plt.figure(2)
plt.scatter(tsne_result_full_328[:, 0], tsne_result_full_328[:, 1], s=3, c='k')

plt.figure(3)
plt.scatter(tsne_result_deci[:, 0], tsne_result_deci[:, 1], s=3, c='k')

plt.figure(5)
plt.scatter(tsne_result_deci_sparce[:, 0], tsne_result_deci_sparce[:, 1], s=3, c='k')

# </editor-fold>
# ----------------------------------------------------------------------------------------------------------------------
