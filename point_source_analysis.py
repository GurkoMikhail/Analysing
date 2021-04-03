import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import zoom

pg.setConfigOptions(imageAxisOrder='row-major')

n_point = 23
distance = 2.5*(n_point + 1)

data = np.load('Processed data/pointSource.npy')
data = np.rot90(data, axes=(1, 2))[n_point]
detector_size = np.array((53.3, 38.7))/100
center = np.array((detector_size[0] + 0.2/100, detector_size[1] - 0.2/100))/2
pixel_size = 0.05/100

pg.mkQApp()
win = pg.GraphicsLayoutWidget()
win.setWindowTitle('Point source analysis')

# A plot area (ViewBox + axes) for displaying the image
p1 = win.addPlot()
p1.setLabel('left', "N")
p1.setLabel('bottom', "X Axis", units='m')
p1.getViewBox().setAspectLocked(detector_size[0]/detector_size[1])

# Item for displaying image data
img = pg.ImageItem()
p1.addItem(img)

# Custom ROI for selecting an image region
roi = pg.CircleROI((-1/100, -1/100), radius=1/100, movable=False, pen='r')
p1.addItem(roi)
roi.setZValue(10)  # make sure ROI is drawn above image

# Contrast/color control
hist = pg.HistogramLUTItem()
hist.setImageItem(img)
win.addItem(hist)

# Another plot area for displaying ROI data
win.nextRow()
p2 = win.addPlot(colspan=2)
p2.setLabel('left', "Y Axis", units='m')
p2.setLabel('bottom', "X Axis", units='m')
p2.setMaximumHeight(250)
win.resize(1600, 900)
win.show()

# Generate image data
img.setImage(data)
hist.setLevels(data.min(), data.max())

# set position and scale of image
img.translate(*(-center))
img.scale(pixel_size, pixel_size)
p1.getAxis('left').setZValue(img.zValue() + 1)
p1.getAxis('bottom').setZValue(img.zValue() + 1)

# zoom to fit imageo
p1.autoRange()  

def culcI(array, interval):
    center = array.size/2
    for i in array[0, center]:
        pass

# Callbacks for handling user interaction
def updatePlot():
    global img, roi, data, p2, p1
    selected = roi.getArrayRegion(data, img)
    # selected = gaussian_filter(selected, 1)
    # center = int(selected.shape[0]/2)
    # selected = selected[(center-3):(center+3), :]
    mean = selected.mean(axis=0)
    # mean_zoom = zoom(mean, 1000, order=1)
    # pos_max = np.argmax(mean_zoom)
    # pos_edge = np.searchsorted(mean_zoom, mean_zoom.max()*0.4)
    # radius = (pos_max - pos_edge)/1000000
    # roi.setSize((radius*2, radius*2), center=(0.5, 0.5))
    p2.plot(x=np.linspace(-mean.size/(2*1000), mean.size/(2*1000), mean.size), y=mean, clear=True)
    p1.setTitle(f'Distance = {distance} cm, diameter = {np.round(roi.size()[0]*1000, 1)} mm')

roi.sigRegionChanged.connect(updatePlot)
updatePlot()

## Start Qt event loop unless running in interactive mode or using pyside.
QtGui.QApplication.instance().exec_()
