import h5py
from pyqtgraph.Qt import QtGui
import numpy as np
from pyqtgraph.graphicsItems.LinearRegionItem import LinearRegionItem
from scipy import ndimage
import pyqtgraph as pg
# import pydicom as dicom
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

# ds = dicom.dcmread('raw')
# print(ds)
# data = ds.pixel_array
# for d in data:
#     print(d.sum())

# file_name = 'Point source 10 sm.hdf'
# file_name = 'efg3_fix left-front projection 5 sm.hdf'
# file_name = 'efg3_fix front projection 2 deg.hdf'
file_name = 'efg3_fix 0.0 deg.hdf'
# file_name = 'efg3_fix front projection 5 sm.hdf'
# file_name = 'efg3_fix_2 front projection 5 sm.hdf'
# file_name = 'efg3_fix front projection 5 sm without collimator.hdf'

file = h5py.File(f'Raw data/{file_name}', 'r')
# file = h5py.File(f'Processed data/{file_name}', 'r')
group = file['Inside Detector']
coordinates = np.copy(group['Coordinates'])
energy_transfer = np.copy(group['Energy transfer'])
emission_time = np.copy(group['Emission time'])
detector_size = np.copy(group['Subject size'])
# detector_size = np.array((50, 40, 40))
file.close()

pixel_size = 0.6
zoom = 10
resolution = 0.4
energy_resolution = 0.099
range_size = np.column_stack([[0, 0], detector_size[:2]])
matrix = (detector_size[:2]/pixel_size).astype(np.int)[:2]

def indices_repeated(emission_time):
    emission_time = np.around(emission_time, decimals=7)
    unique, unique_counts = np.unique(emission_time, return_counts=True)
    indices = np.nonzero(unique_counts > 1)[0]
    unique = unique[indices]
    indices = []
    for time in unique:
        indices.append(np.nonzero(emission_time == time)[0])
    return indices

def average_coordinates(coordinates, energy, emission_time):
    indices = indices_repeated(emission_time)
    del_indices = []
    for event_indices in indices:
        weights = energy[event_indices]
        energy[event_indices[0]] = np.sum(weights)
        average = np.average(coordinates[event_indices], axis=0, weights=weights)
        coordinates[event_indices[0]] = average
        del_indices.append(event_indices[1:])
    del_indices = np.concatenate(del_indices)
    coordinates = np.delete(coordinates, del_indices)
    energy = np.delete(energy, del_indices)
    emission_time = np.delete(emission_time, del_indices)

def energy_deviation(energy, coeff):
    resolution_distribution = coeff/np.sqrt(energy/10**6)
    sigma = resolution_distribution*energy/2.355
    energy[:] = np.random.normal(energy, sigma)

def coordinates_deviation(coordinates, resolution):
    sigma = resolution/2.35
    coordinates[:] = np.random.normal(coordinates, sigma)

def enough_energy(min_energy, max_energy):
    # return energy_transfer == 140500
    return (energy_transfer >= min_energy)*(energy_transfer <= max_energy)

def updateImage():
    global image, energy_region, coordinates, resolution, pixel_size, matrix, hist
    min_energy, max_energy = energy_region.getRegion()
    indices = np.nonzero(enough_energy(min_energy, max_energy))[0]
    # indices = indices[:20000]
    print(f'Energy window: {(int(min_energy), int(max_energy))}')
    print(f'Counts inside window = {indices.size}')
    histogram = np.histogram2d(coordinates[indices, 0], coordinates[indices, 1], bins=matrix, range=range_size[:2])
    data = histogram[0]
    data = ndimage.zoom(data, zoom, order=1)
    image.setImage(data)
    hist.setLevels(data.min(), data.max())
    # hist.setLevels(0, 270)

average_coordinates(coordinates, energy_transfer, emission_time)
coeff = np.sqrt(0.14)*energy_resolution
energy_deviation(energy_transfer, coeff)
coordinates_deviation(coordinates, resolution)

pg.mkQApp()
win = pg.GraphicsLayoutWidget()
win.setWindowTitle(file_name)

p1 = win.addPlot()
p1.setTitle(f'matrix = {matrix}, pixel size = {round(pixel_size*10, 2)} mm, resolution = {round(resolution*10, 2)} mm')
p1.setLabel('left', "Y Axis", units='m')
p1.setLabel('bottom', "X Axis", units='m')
p1.getViewBox().setAspectLocked(detector_size[0]/detector_size[1])

image = pg.ImageItem()
p1.addItem(image)
p1.getAxis('left').setZValue(image.zValue() + 1)
p1.getAxis('bottom').setZValue(image.zValue() + 1)
# image.translate(*(-detector_size[:2]/100/2))
image.scale(0.01/zoom, 0.01/zoom)
# p1.showGrid(True, True)
image.scale(pixel_size, pixel_size)

hist = pg.HistogramLUTItem()
hist.setImageItem(image)
win.addItem(hist)

win.nextRow()
p2 = win.addPlot(colspan=2, title='Energy spectrum')
p2.setLabel('left', 'N')
p2.setLabel('bottom', 'Energy', units='eV')
p2.setLogMode(y=True)
p2.showGrid(x=True, y=True)
p2.setMaximumHeight(250)
win.resize(1200, 1000)
win.show()
energy_distribution, energy = np.histogram(energy_transfer, bins=1024)
energy_distribution += 1

p2.plot(x=energy[1:], y=energy_distribution, clear=True).setPen((0, 0, 0, 255))


energy_region = LinearRegionItem((126*10**3, 154*10**3))
p2.addItem(energy_region)
energy_region.sigRegionChangeFinished.connect(updateImage)

updateImage()

# del coordinates, energy_transfer, emission_time, emission_coordinates

QtGui.QApplication.instance().exec_()

