

from . import io_with_cpp as io
from . import gpu
import matplotlib.pylab as pylab
from subprocess import Popen, PIPE
import sys
from os.path import join

def t_sne(samples, files_dir=None, exe_dir=None, num_dims=2, perplexity=100, theta=0.4, eta=200, iterations=1000,
          random_seed=-1, verbose=2):

    data = pylab.demean(samples, axis=0)
    data /= data.max()
    closest_indices_in_hd, closest_distances_in_hd = \
        gpu.calculate_knn_distances(template_features_sparse_clean=data, perplexity=perplexity,
                                               mem_usage=0.9,
                                               verbose=True)

    io.save_data_for_barneshut(files_dir, closest_distances_in_hd, closest_indices_in_hd, eta=eta, iterations=iterations,
                               num_of_dims=num_dims, perplexity=perplexity, theta=theta, random_seed=random_seed,
                               verbose=verbose)
    del samples

    # Call Barnes_Hut and let it do its thing
    if exe_dir is None:
        exe_file = io.find_exe_file()
        if exe_file is None:
            print('Cannot find Barnes_Hut.exe. Please provide a path to it yourself by setting the the exe_dir parameter.')
            return
    else:
        exe_file = join(exe_dir, 'Barnes_Hut.exe')

    with Popen(['Barnes_Hut.exe', ], executable=exe_file, cwd=files_dir, stdout=PIPE, bufsize=1, universal_newlines=True) \
            as t_sne_bhcuda_p:
        for line in iter(t_sne_bhcuda_p.stdout):
            print(line, end='')
            sys.stdout.flush()
        t_sne_bhcuda_p.wait()
    assert not t_sne_bhcuda_p.returncode, ('ERROR: Call to Barnes_Hut exited '
                                           'with a non-zero return code exit status, please ' +
                                           ('enable verbose mode and ' if not verbose else '') +
                                           'refer to the t_sne output for further details')

    tsne = io.load_tsne_result(files_dir)

    return tsne
