__author__ = 'IntelligentSystem'


def gdft_plot_topo(channelsPositions, data, **kwargs):
    if not kwargs.get('hpos'):
        hpos = 0
    else:
        hpos = kwargs['hpos']
    if not kwargs.get('vpos'):
        vpos = 0
    else:
        vpos = kwargs['vpos']
    if not kwargs.get('width'):
        width = None
    else:
        width = kwargs['width']
    if not kwargs.get('height'):
        height = None
    else:
        height = kwargs['height']
    if not kwargs.get('gridscale'):
        gridscale = 67
    else:
        gridscale = kwargs['gridscale']
    if not kwargs.get('shading'):
        shading = "flat"
    else:
        shading = kwargs['shading']
#    if not kwargs.get('interplim'):
#        interplim = "electrodes"    #"electrodes" OR "mask"
#    else:
#        interplim = kwargs['interplim']
    if not kwargs.get('interpmethod'):
        interpmethod = "cubic"      #"nearest", "linear", "cubic", "spline"
    else:
        interpmethod = kwargs['interpmethod']
    if not kwargs.get('style'):
        style = "surfiso"
    else:
        style = kwargs['style']
#    if not kwargs.get('datmask'):
#        datmask = None
#    else:
#        datmask = kwargs['datmask']
#    if not kwargs.get('mask'):
#        mask = None
#    else:
#        mask = kwargs['mask']
    if not kwargs.get('outline'):
        outline = None
    else:
        outline = kwargs['outline']

    if np.isnan(data).any():
        warnings.warn('The data passed to gdft_plot_topo contain NaN values. These will create unexpected results in the interpolation. Deal with them.')

    channelsPositions = np.array(channelsPositions)
    allCoordinates = channelsPositions
#    if mask:
#        allCoordinates = [channelsPositions, mask]
    if outline:
        allCoordinates = [channelsPositions, outline]

    naturalWidth = np.max(allCoordinates[:,0]) - np.min(allCoordinates[:,0])
    naturalHeight = np.max(allCoordinates[:,1]) - np.min(allCoordinates[:,1])

    if not width and not height:
        xScaling = 1;
        yScaling = 1;
    elif not width and height:
        yScaling = height/naturalHeight
        xScaling = yScaling
    elif width and not height:
        xScaling = width/naturalWidth
        yScaling = xScaling
    elif width and height:
        xScaling = width/naturalWidth
        yScaling = height/naturalHeight

    #channelsPositionOriginal = channelsPositions
    chanX = channelsPositions[:,0] * xScaling + hpos
    chanY = channelsPositions[:,1] * yScaling + vpos

    ##Check if all channel coordinates are whithin the layout mask. If not adjust the mask accordingly
    #newpoints = np.array(shape(1))
    #outside = np.ones(len(chanX))<0
    #inside = gdft_inside_contour(channelsPositions, mask)
    #outside = np.logical_not(inside)
    #newpoints = [chanX[outside], chanY[outside]]

#    if interplim=='electrodes':
    hlim = [np.min(chanX), np.max(chanX)]
    vlim = [np.min(chanY), np.max(chanY)]
#    elif interplim=='mask' and mask:
#        hlim = vlim = [-np.Inf, np.Inf]
#        hlim = [min(hlim[0], min(mask[:,0]*xScaling+hpos)), max(hlim[1], min(mask[:,1]*xScaling+hpos))]
#        vlim = [min(vlim[0], min(mask[:,0]*yScaling+vpos)), max(vlim[1], min(mask[:,1]*yScaling+vpos))]
#    else:
#       hlim = [np.min(chanX), np.max(chanX)]
#       vlim = [np.min(chanY), np.max(chanY)]
#
#    if mask:
#       maskimage = np.logical_not(np.ones([gridscale,gridscale]))
#       xi = np.linspace(hlim[0], hlim[1], gridscale)
#       yi = np.linspace(vlim[0], vlim[1], gridscale)
#       [Xi, Yi] = np.meshgrid(xi, yi)
#       maskimage[gdft_inside_contour([Xi,Yi], mask)] = True
#    else:
#       maskimage = np.array(np.shape(1))

#    if datmask:

    xi, yi = np.mgrid[hlim[0]:hlim[1]:complex(0,gridscale), vlim[0]:vlim[1]:complex(0,gridscale)]

    Zi = interpolate.griddata((chanX,chanY), data,  (xi,yi), interpmethod)


    dataPlot = mmlab.imshow(Zi)
    dataPlot.actor.scale = [(hlim[1])/(gridscale-1), (vlim[1])/(gridscale-1), 1]
    dataPlot.actor.position = [(hlim[1]+hlim[0])/2, (vlim[1]+vlim[0])/2, 0]
    chanPlot = mmlab.points3d(chanX, chanY, np.zeros(np.shape(chanX)))
    chanPlot.glyph.glyph.scale_factor = 20
   # mmlab.pipeline.ContourGridPlaneFactory(dataPlot)
    #mmlab.pipeline.iso_surface()





def gdft_inside_contour(channelsPositions, contour):
    npos = np.shape(channelsPositions)[0]
    ncnt = np.int(np.shape(contour)[0])
    x= channelsPositions[:,0]
    y= channelsPositions[:,1]

    minx = min(x)
    miny = min(y)
    maxx = max(x)
    maxy = max(y)

    inOut = (np.ones(npos)>0)
    inOut[x<minx] = False
    inOut[x>maxx] = False
    inOut[y<miny] = False
    inOut[y>maxy] = False

    sel = np.squeeze(np.where(inOut))
    for i in range(0,len(sel)):
        inOut[sel[i]] = pointInPoly(ncnt, contour[:,0], contour[:,1], channelsPositions[i,0], channelsPositions[i,1])

    return inOut


def pointInPoly(nvert, vertx, verty, testx, testy):
    """
    nvert = number of vertices in the contour
    vertx = array of the x coordinates of the vertices
    verty = array of the y coordinates of the vertices
    testx = x coordinate of the point
    testy = y coordinate of the point
    """
    c = False
    z = nvert - 1
    for i in range(0,nvert-1):
        if  ((verty[i]>testy) != (verty[z]>testy)) and (testx < (vertx[z]-vertx[i]) * (testy-verty[i]) / (verty[z]-verty[i]) + vertx[i]):
            c = not c
        z = i
    return c