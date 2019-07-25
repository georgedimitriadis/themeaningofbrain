
import matplotlib.pyplot as plt
import sys
from io import StringIO
import numpy as np


# ---------------------------
# USE THE FOLLOWING TO INTERACTIVELY SEE THE TEMPLATE TIME PLOT WHEN CLICKING ON A POINT ON A TEMPLATE POSITIONS PLOT
# Use as follows:

# template_info = np.load(join(kilosort_folder, 'template_info.df'))
# avg_templates = np.load(join(kilosort_folder, 'avg_spike_template.npy'))
#
# f = plt.figure(0)
# old_stdout = pl_funcs.setup_for_capturing_template_positions_from_image()
#
# show_average_template = pl_funcs.show_average_template
# template_number = 0
#
# tr.connect_repl_var(globals(), 'f', 'template_number', 'show_average_template')
# ---------------------------

def setup_for_capturing_template_positions_from_image():
    old_stdout = sys.stdout
    global previous_template_number
    previous_template_number = -1
    global result
    result = StringIO()

    return old_stdout


def show_average_template(figure):
    global previous_template_number
    global result
    sys.stdout = result
    string = result.getvalue()
    new = string[-200:]
    try:
        template_number = int(new[new.find('Template number'): new.find('Template number')+22][18:22])
        if template_number != previous_template_number:
            template = template_info[template_info['template number'] == template_number]
            figure.clear()
            ax = figure.add_subplot(111)
            try:
                ax.plot(np.squeeze(avg_templates[template.index.values]).T)
            except:
                pass
        previous_template_number = template_number
        figure.suptitle('Template = {}, with {} number of spikes'.format(str(template_number),
                                                                         str(template['number of spikes'].values[0])))
    except:
        template_number = None
    return template_number

# ---------------------------