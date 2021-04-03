import numpy as np
from time import time
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.functions import makeRGBA
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import h5py

from PyQt5 import uic


pg.mkQApp() 

# QtGui.QApplication.exec()
# win = uic.loadUi("UI/test.ui")
# win.show()
# # sys.exit(app.exec())
# spaceView = win.graphicsView

name = 'efg3_fix front projection'

space_size = np.asarray((51.2, 58.2, 51.2))

spaceView = gl.GLViewWidget()
spaceView.setCameraPosition(distance=np.max(space_size)*3)
spaceView.pan(*space_size/2)
# spaceView.setTitle('Visualisation')
spaceView.setGeometry(0, 110, 1920, 1080)
volume = gl.GLScatterPlotItem()

coordinates = []
energy_transfer = []
emission_coordinates = []

file = h5py.File(f'Output data/{name}', 'r')
# group = file['Flows/(0.0, 0.01)']
# del group
times = np.asarray(sorted([list(map(float, key[1:-1].split(','))) for key in file['Flows'].keys()], key=lambda time: time[1]))
print(times.max())

for flow in file['Flows'].values():
    coordinates.append(flow['Coordinates'])
    energy_transfer.append(flow['Energy transfer'])
    emission_coordinates.append(flow['Emission coordinates'])
coordinates = np.concatenate(coordinates)
energy_transfer = np.concatenate(energy_transfer)
emission_coordinates = np.concatenate(emission_coordinates)
file.close()

# for flow in times[-10:]:
#     flow = file['Flows'][f'{tuple(flow)}']
#     coordinates.append(flow['Coordinates'])
#     energy_transfer.append(flow['Energy transfer'])
#     emission_coordinates.append(flow['Emission coordinates'])
# coordinates = np.concatenate(coordinates)
# energy_transfer = np.concatenate(energy_transfer)
# emission_coordinates = np.concatenate(emission_coordinates)
# file.close()

def update_data():
    enough_energy = (energy_transfer >=120*10**3)*(energy_transfer <= 140*10**3)
    emission_point = (emission_coordinates[:, 1] >= 0)*(emission_coordinates[:, 1] <= 58.2)
    # emission_point *= (coordinates[:, 0] >= 0)*(coordinates[:, 0] <= 3)
    # emission_point *= (coordinates[:, 2] >= 10)*(coordinates[:, 2] <= 30)
    indices = np.nonzero(enough_energy*emission_point)[0]
    volume.setData(pos=coordinates[indices], size=1, color=(0, 50, 255, 255))
    return indices

update_data()

space_box = gl.GLBoxItem()
space_box.setSize(*space_size)
spaceView.addItem(space_box)

# phantom_volume = np.load('Phantoms/efg3cut_phantom.npy')
phantom_volume = np.histogramdd(
    coordinates,
    bins=(space_size/0.4).astype(np.int),
    range=((0, space_size[0]), (0, space_size[1]), (0, space_size[2])),
    weights=energy_transfer
)[0]

del coordinates, energy_transfer, emission_coordinates

levels = [
    np.min(phantom_volume),
    np.max(phantom_volume)*0.05
]

zeros = np.nonzero(phantom_volume < levels[1]*0.1)
phantom_volume[zeros] = np.NaN

phantom_RGBA = []
for slice in phantom_volume:
    rgba, alpha = makeRGBA(data=slice, levels=levels)
    rgba[:, :, 3] = rgba[:, :, 3]/20
    # indices = np.nonzero(rgba[:, :, 0] + rgba[:, :, 1] + rgba[:, :, 2])
    # rgba[indices[0], indices[1], 3] = 10
    # rgba[:, :, 3] = rgba[:, :, 3]*(rgba[:, :, 0]/255 + rgba[:, :, 1]/255 + rgba[:, :, 2]/255)
    phantom_RGBA.append(rgba)
phantom_RGBA = np.asarray(phantom_RGBA)
phantom_volume = phantom_RGBA
phantom_volume = gl.GLVolumeItem(phantom_volume, sliceDensity=1)
# phantom_volume.translate(0., 7., 0.)
phantom_volume.scale(0.4, 0.4, 0.4, local=True)
spaceView.addItem(phantom_volume)

collimator_size = np.array((51.2, 51.2, 3.5))

collimator_box = gl.GLBoxItem(color=(255, 0, 0, 255))
collimator_box.setDepthValue(-300)
collimator_box.setSize(*collimator_size)
collimator_box.rotate(90, 1, 0, 0)
collimator_box.translate(0, 7.0, 0)
spaceView.addItem(collimator_box)

detector_size = np.array((51.2, 51.2, 3.0))

detector_box = gl.GLBoxItem(color=(0, 255, 0, 255))
detector_box.setDepthValue(-300)
detector_box.setSize(*detector_size)
detector_box.rotate(90, 1, 0, 0)
detector_box.translate(0, 3., 0)
spaceView.addItem(detector_box)

space_axis = gl.GLAxisItem()
space_axis.setSize(*(space_size + 10.0))
space_axis.setDepthValue(-200)
spaceView.addItem(space_axis)

# spaceView.addItem(volume)
spaceView.show()
QtGui.QApplication.exec()

