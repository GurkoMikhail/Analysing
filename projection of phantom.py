from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy as np
from scipy.ndimage.filters import gaussian_filter

pixel_size = 0.4
resolution = 0.75/2.355

volume = np.load('Phantoms/efg3_fix.npy')
print(np.unique(volume))
data = np.sum(volume, axis=1)
# data = data[::-1]
data = data[:, :100]

pg.mkQApp()
win = pg.GraphicsLayoutWidget()
win.setWindowTitle(f'Projection of phantom')

p1 = win.addPlot()
p1.setLabel('left', "Y Axis", units='sm')
p1.setLabel('bottom', "X Axis", units='sm')

image = pg.ImageItem()
p1.addItem(image)
image.scale(pixel_size, pixel_size)

hist = pg.HistogramLUTItem()
hist.setImageItem(image)
win.addItem(hist)

data = gaussian_filter(data, (resolution/pixel_size, resolution/pixel_size))
image.setImage(data)

# win.nextRow()
# p2 = win.addPlot(colspan=2, title='Energy distribution')
# p2.setLabel('left', 'N')
# p2.setLabel('bottom', 'Energy', units='eV')
# # p2.setLogMode(y=True)
# p2.setMaximumHeight(250)
win.resize(1200, 1000)
win.show()


# rng_free_path = np.random.default_rng()
# max_lac = 0.2
# free_path = rng_free_path.exponential(1/max_lac, 10**6)
# print(free_path.mean())
# free_path_distribution, free_path = np.histogram(free_path, bins=512, density=False)
# p2.plot(x=free_path[1:], y=free_path_distribution, clear=True)

# minEnergyLine = pg.InfiniteLine(120*10**3, movable=True, markers=[('^', 0), ('v', 1)], pen='y')
# p2.addItem(minEnergyLine)
# minEnergyLine.sigPositionChangeFinished.connect(updateImage)

# maxEnergyLine = pg.InfiniteLine(160*10**3, movable=True, markers=[('^', 0), ('v', 1)], pen='y')
# p2.addItem(maxEnergyLine)
# maxEnergyLine.sigPositionChangeFinished.connect(updateImage)

# updateImage()

# del coordinates, energy_transfer, emission_time, emission_coordinates

win.resize(1200, 1000)
win.show()

QtGui.QApplication.instance().exec_()

