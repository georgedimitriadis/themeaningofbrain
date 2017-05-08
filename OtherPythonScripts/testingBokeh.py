from random import random
from bokeh.models import CustomJS, ColumnDataSource
from bokeh.plotting import hplot, figure, output_file, show
from OtherPythonScripts import TestingStuff

output_file("callback.html")

x = [random() for x in range(500)]
y = [random() for y in range(500)]

s1 = ColumnDataSource(data=dict(x=x, y=y))
p1 = figure(plot_width=400, plot_height=400, tools="lasso_select", title="Select Here")
p1.circle('x', 'y', source=s1, alpha=0.6)

s2 = ColumnDataSource(data=dict(x=[], y=[]))
p2 = figure(plot_width=400, plot_height=400, x_range=(0, 1), y_range=(0, 1),
            tools="", title="Watch Here")
p2.circle('x', 'y', source=s2, alpha=0.6)

def callback(source=s2):
    inds = cb_obj.selected['1d'].indices
    d1 = cb_obj.get('data')
    d2 = source.get('data')
    d2['x'] = []
    d2['y'] = []
    for i in range(inds.length):
        d2['x'].push(d1['x'][inds[i]])
        d2['y'].push(d1['y'][inds[i]])
    s2.trigger('change')

s1.callback = CustomJS.from_py_func(callback)


layout = hplot(p1, p2)

show(layout)



s1.callback = CustomJS(args=dict(s2=s2), code="""
        var inds = cb_obj.get('selected')['1d'].indices;
        var d1 = cb_obj.get('data');
        var d2 = s2.get('data');
        d2['x'] = []
        d2['y'] = []
        for (i = 0; i < inds.length; i++) {
            d2['x'].push(d1['x'][inds[i]])
            d2['y'].push(d1['y'][inds[i]])
        }
        s2.trigger('change');
    """)











from bokeh.io import vform
from bokeh.models import CustomJS, ColumnDataSource, Slider
from bokeh.plotting import Figure, output_file, show

output_file("callback.html")

x = [x*0.005 for x in range(0, 200)]
y = x

source = ColumnDataSource(data=dict(x=x, y=y))

plot = Figure(plot_width=400, plot_height=400)
plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)



slider = Slider(start=0.1, end=4, value=1, step=.1, title="power",
                callback=CustomJS.from_py_func(TestingStuff.callback(source=source)))

layout = vform(slider, plot)

show(layout)