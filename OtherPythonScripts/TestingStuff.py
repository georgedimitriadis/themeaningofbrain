


import numpy as np

from bokeh.plotting import figure, show, output_file

N = 4000
x = np.random.random(size=N) * 100
y = np.random.random(size=N) * 100
radii = 0.5
colors = [
    "#%02x%02x%02x" % (int(r), 200, 50) for r in 2.5*x
]

TOOLS="resize,crosshair,pan,wheel_zoom,box_zoom,undo,redo,reset,tap,previewsave,box_select,poly_select,lasso_select"

p = figure(tools=TOOLS)

p.scatter(x, y, radius=radii,
          fill_color=colors, fill_alpha=1,
          line_color=None)

output_file("color_scatter.html", title="color_scatter.py example")

show(p)  # open a browser



from t_sne_bhcuda import spike_heatmap, tsne_cluster
cluster_info = tsne_cluster.load_cluster_info(cluster_info_filename)





spikes = tsne_cluster.load_cluster_info(cluster_info_filename)['Spike_Indices']['Juxta']




from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource

shape_of_spike_trig_avg = ((num_ivm_channels,
                            num_of_points_in_spike_trig,
                            20000))
cut_extracellular_data = tsne_cluster.load_extracellular_data_cube(data_cube_filename, cube_type,
                                                                   shape_of_spike_trig_avg)
spikes = np.arange(100)

prb_file = r'D:\Data\George\Projects\SpikeSorting\Joana_Paired_128ch\2015-09-03\Analysis\klustakwik\128ch_passive_imec.prb'

prb_file = r'D:\Data\George\Projects\SpikeSorting\Joana_Paired_128ch\2015-09-03\Analysis\klustakwik' + \
           r'\128ch_passive_imec_madeup_2shank.prb'


final_image, (x_size, y_size) = spike_heatmap.create_heatmap(cut_extracellular_data[:, :, spikes], prb_file,
                                                             voltage_step_size=1e-6, scale_microvolts=1000000,
                                                             window_size=60,
                                                             rotate_90=True, flip_ud=True, flip_lr=False)

plot_height = 800
plot_width = max(200, int(plot_height * y_size / x_size))

heatmap_plot = figure(plot_width=plot_width, plot_height=plot_height, x_range=(0, x_size), y_range=(0, y_size))

heatmap_data_source = ColumnDataSource(data=dict(image=[final_image], x=[0], y=[0], dw=[x_size], dh=[y_size]))
heatmap_renderer = heatmap_plot.image_rgba(source=heatmap_data_source, image='image', x='x', y='y',
                                           dw='dw', dh='dh', dilate=False)

heatmap_plot.axis.visible = None
heatmap_plot.xgrid.grid_line_color = None
heatmap_plot.ygrid.grid_line_color = None

output_file("image_rgba.html", title="image_rgba.py example")

show(heatmap_plot)  # open a browser




from bokeh.models.widgets import CheckboxGroup
from bokeh.io import output_file, show, vform

output_file("checkbox_group.html")

checkbox_group = CheckboxGroup(
        labels=["Option 1"], active=[0])

def update(attr, old, new):
    print(new.active)

checkbox_group

show(vform(checkbox_group))






from bokeh.models.widgets import DataTable, TableColumn, Button, Select, Toggle

from bokeh.plotting import *
import numpy as np

N = 9
x = np.linspace(-2, 2, N)
y = x**2

output_file("glyphs.html", title="glyphs.py example")

color_list = np.array(('#d18096', '#483496', '#00FFD0'))
labels = [0,1,2,0,1,0,0,0,2]

# Materialize the full list of colors, using Numpy's "fancy indexing"
# (which is why we needed to create color_list as a numpy array)
colors = color_list[labels]
b = Toggle(type='warning')
f = Figure()
s = f.scatter(x, y, size=16,
        fill_color=colors, fill_alpha=0.6, alpha=0.4)
#s.data_source.data = {'x':[],'y':[]}
s = f.scatter(y, x, size=20,
              fill_color=colors, fill_alpha=0.6, alpha=0.4)
show(b)