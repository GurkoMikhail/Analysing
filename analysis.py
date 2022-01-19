import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

# Interpret image data as row-major instead of col-major
# pg.setConfigOptions(imageAxisOrder='row-major')

pg.mkQApp()
win = pg.GraphicsLayoutWidget()
win.setWindowTitle('Image Analysis')

# A plot area (ViewBox + axes) for displaying the image
p1 = win.addPlot(title="")
p1.setLabel('left', text='Z', units='pixel')
p1.setLabel('bottom', text='Y', units='pixel')
p1.setAspectLocked()

# Item for displaying image data
img = pg.ImageItem()
p1.addItem(img)

# Custom ROI for selecting an image region
roi = pg.ROI([0, 0], scaleSnap=True, translateSnap=True)
roi.addScaleHandle([0.5, 1], [0.5, 0.5])
roi.addScaleHandle([0, 0.5], [0.5, 0.5])
# p1.addItem(roi)
roi.setZValue(10)  # make sure ROI is drawn above image

# Isocurve drawing
iso = pg.IsocurveItem(level=20, pen='g')
# iso.setParentItem(img)
iso.setZValue(5)

# Contrast/color control
hist = pg.HistogramLUTItem()
hist.setImageItem(img)
hist.setMaximumWidth(100)
win.addItem(hist)

# Draggable line for setting isocurve level
isoLine = pg.InfiniteLine(angle=0, movable=True, pen='g')
hist.vb.addItem(isoLine)
hist.vb.setMouseEnabled(y=False) # makes user interaction a little easier
isoLine.setValue(20)
isoLine.setZValue(1000) # bring iso line above contrast controls

# Another plot area for displaying ROI data
win.nextRow()
p2 = win.addPlot(colspan=1)
# p2.setLabel('left', text='Fraction of scattered', units='%')
p2.setLabel('left', text='Counts')
p2.setLabel('bottom', text='Z', units='pixel')
p2.showGrid(x=True, y=True, alpha=0.7)
p2.setMaximumHeight(250)
win.resize(1000, 1000)
win.show()


# Load image data
phantom_name = 'ae3'
n = 23
# phantom_name = 'efg3_LA_liver'
# phantom_name = phantom_name + '_scatters'
# phantom_name = phantom_name + '_140500eV'

# data = np.load('Numpy data/efg3_32.npy')
# data = np.loadtxt('Dat phantoms/ae3.dat').reshape((128, 128, 128), order='F')

# data3d = np.load(f'Numpy data/{phantom_name}.npy')
# data = data3d.sum(axis=0)
# data = np.rot90(data, k=-1, axes=(0, 2))
# image = data.sum(axis=0)

data = np.loadtxt(f'Dat phantoms/{phantom_name}.dat').reshape((128, 128, 128), order='F')
# lung = np.where(data == 0.04, data, 0)
# data
# image = data[:, :, 48]


image = data.sum(axis=0)


# p1.setTitle(phantom_name)
# data = np.load(f'Numpy data/{phantom_name}.npy')
# image = data[n]
# image = np.loadtxt(f'Dat data/{phantom_name}/{n}.dat')
# data *= 100
# print(f'Total {image.sum()}')
# print(f'Mean {image[image > 0].mean()}')
sigma = 0
image = gaussian_filter(image, sigma)
pixelSize = 0.5*10**(-2)
print(image.sum())
img.setImage(image)
roi.setPos(0, (image.shape[0] - 1)//2 - 2)
roi.setSize((image.shape[1], 4))
roi.maxBounds = img.boundingRect()
# hist.setLevels(data.min(), data.max())

# build isocurves from smoothed data
iso.setData(gaussian_filter(image, 0 if sigma != 0 else 1))

# set position and scale of image
# img.scale(0.2, 0.2)
# img.translate(-50, 0)

# zoom to fit imageo
# p1.autoRange()  


# Callbacks for handling user interaction
total = image.sum()
mean = image[image > 0].mean()
def updatePlot():
    global img, roi, image, p2, data
    y = data.sum(axis=(0, 1))
    x = np.arange(y.size) + 1
    p2.plot(x=x, y=y, clear=True).setPen('b')
    # selected = roi.getArrayRegion(image, img)
    # # print(f'ROI mean {selected[selected > 0].mean()}')
    # p2.plot(selected.mean(axis=0), clear=True).setPen('b')
    # roi_mean = selected[selected > 0].mean()
    # p2.setTitle('Total = %i, mean = %0.1f, ROI mean = %0.1f' %(total, mean, roi_mean))

roi.sigRegionChanged.connect(updatePlot)
updatePlot()

def updateIsocurve():
    global isoLine, iso
    iso.setLevel(isoLine.value())

isoLine.sigDragged.connect(updateIsocurve)

def imageHoverEvent(event):
    """Show the position, pixel, and value under the mouse cursor.
    """
    total = image.sum()
    mean = image[image > 0].mean()
    if event.isExit():
        p1.setLabel('top', "Total: %i, Mean: %d" % (total, mean))
        return
    pos = event.pos()
    i, j = pos.y(), pos.x()
    i = int(np.clip(i, 0, image.shape[0] - 1))
    j = int(np.clip(j, 0, image.shape[1] - 1))
    val = image[i, j]
    ppos = img.mapToParent(pos)
    x, y = ppos.x(), ppos.y()
    p1.setLabel('top', "pos: (%0.1f, %0.1f)  pixel: (%d, %d)  value: %g Total: %i, Mean: %d" % (x, y, i, j, val, total, mean))

# Monkey-patch the image to use our custom hover function. 
# This is generally discouraged (you should subclass ImageItem instead),
# but it works for a very simple use like this. 
# img.hoverEvent = imageHoverEvent


## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
