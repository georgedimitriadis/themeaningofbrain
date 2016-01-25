__author__ = 'George Dimitriadis'


import numpy as np


def _generate_adjacency_graph(all_electrodes):
    graph_dict = {}
    for r in np.arange(all_electrodes.shape[0]):
        for c in np.arange(all_electrodes.shape[1]):
            electrode = all_electrodes[r, c]
            if electrode != -1:
                graph_dict[electrode] = []
                for step_r in np.arange(-1, 2):
                    if -1 < r + step_r < all_electrodes.shape[0]:
                        for step_c in np.arange(-1, 2):
                            if -1 < c + step_c < all_electrodes.shape[1] and not (step_r ==0 and step_c == 0):
                                neighbour = all_electrodes[r + step_r, c + step_c]
                                if neighbour != -1:
                                    try:
                                        graph_dict[neighbour]
                                    except:
                                        graph_dict[electrode].append(all_electrodes[r + step_r, c + step_c])
                                    else:
                                        try:
                                            graph_dict[neighbour].index(electrode)
                                        except:
                                            graph_dict[electrode].append(all_electrodes[r + step_r, c + step_c])
                if len(graph_dict[electrode]) == 0:
                    graph_dict.pop(electrode)

    return graph_dict



def generate_prb_file(filename, all_electrodes_array):

    graph_dict = _generate_adjacency_graph(all_electrodes_array)

    file = open(filename, 'w')
    file.write('channel_groups = {\n')
    file.write('    # Shank index.\n')
    file.write('    0:\n')
    file.write('        {\n')
    file.write('            # List of channels to keep for spike detection.\n')
    file.write('            \'channels\': list(range({})),\n'.format(128))
    file.write('\n')
    file.write('            # Adjacency graph. Dead channels will be automatically discarded\n')
    file.write('            # by considering the corresponding subgraph.\n')
    file.write('            \'graph\': [\n')
    for key in graph_dict.keys():
        line = '                '
        for neighbour in graph_dict[key]:
            line = line + '({}, {}),'.format(key, neighbour)
        file.write(line + '\n')
    file.write('            ],\n')
    file.write('\n')
    file.write('            # 2D positions of the channels, only for visualization purposes\n')
    file.write('            # in KlustaViewa. The unit doesn\'t matter.\n')
    file.write('            \'geometry\': {\n')
    step = 10
    for r in np.arange(all_electrodes_array.shape[0]):
        for c in np.arange(all_electrodes_array.shape[1]):
            electrode = all_electrodes_array[r, c]
            if electrode != -1:
                file.write('                {}: ({}, {}),\n'.format(electrode, r * 10, c * 10))
    file.write('            }\n')
    file.write('    }\n')
    file.write('}\n')
    file.close()
