from PyQt5 import QtGui
from PyQt5.uic import loadUi
from numpy.lib.function_base import rot90
from pyqtgraph.metaarray.MetaArray import axis
from scipy.ndimage import zoom, gaussian_filter
from numpy import array, asarray, asfortranarray, loadtxt, min, max, load, log, savetxt, sqrt, linspace, unique, zeros, float64, absolute, zeros_like, int64, pi, histogramdd
from pyqtgraph.Qt import mkQApp
from pyqtgraph import GradientEditorItem, makeRGBA
from pyqtgraph.opengl import GLBoxItem, GLVolumeItem, GLAxisItem
from time import sleep
import h5py

# file_name = 'point_source 10.0 cm.hdf'
# file_name = 'LEHR.hdf'
# # file_name = 'Attenuation test Water.hdf'
# # # file_name = 'Attenuation inner test Bone.hdf'
# file_name = 'efg3_5deg_solid_angle 0.0 deg.hdf'
# # # file_name = 'efg3_fix front projection.hdf'
# # # file_name = 'efg3_fix front projection 5 sm without collimator.hdf'

# file = h5py.File(f'Processed volume/{file_name}')
# file = h5py.File(f'Raw data/{file_name}')
# distribution = file['Dose distribution']
# distribution = file['Emission distribution']
# volume = array(distribution['Volume'])
# volume = sqrt(volume)
# volume = log(volume + 2)
# voxel_size = array(distribution['Voxel size'])
# space_parameters = file['Modeling parameters/Space']

# collimator_size = array(space_parameters['Collimator/size'])
# collimator_coordinates = array(space_parameters['Collimator/coordinates'])
# collimator_angles = array(space_parameters['Collimator/euler_angles'])

# detector_size = array(space_parameters['Detector/size'])
# detector_coordinates = array(space_parameters['Detector/coordinates'])
# detector_angles = array(space_parameters['Detector/euler_angles'])

# file.close()
# volume = load('Numpy phantoms/efg3.npy')
# volume = load('Phantoms/efg3.npy')
# volume = rot90(load('Numpy phantoms/efg3_changed.npy'), k=-1, axes=(1, 2))
# volume /= volume.sum()
# volume = rot90(load('Numpy phantoms/efg3.npy'))[:, ::-1]
# volume = asfortranarray(rot90(volume, k=-1, axes=(1, 2)))
# volume = volume[8:108]
# volume = rot90(volume, k=-1, axes=(1, 2))

# volume = loadtxt('Dat phantoms/fgr3-AC_1_99.dat').reshape((128, 128, 100), order='F')

# volume1 = loadtxt('Dat phantoms/fgr3_WS_nonAC.dat').reshape((128, 128, 100), order='F')
# volume1 /= volume1.sum()
# volume2 = loadtxt('Dat phantoms/fgr3_WS_AC.dat').reshape((128, 128, 100), order='F')
# volume2 /= volume2.sum()
# volume = abs(volume1 - volume2)
# volume = volume1*volume2

# savetxt('Dat phantoms/fgr3_WS_normDiff.dat', volume.ravel(order='F'), fmt='\t%.6E')

# volume *= loadtxt('Dat phantoms/ae.dat').reshape((128, 128, 100), order='F')
# volume *= volume_2.sum()
# volume -= volume_2
# volume = abs(volume)
# heart_slice = array((
#     (59, 85),
#     (40, 72),
#     (51, 74)
#     ))

# volume = zoom(volume, 1/2, order=5)
# volume = load('Phantoms/efg3_new.npy')
# volume = load('Raw data/coordinates.npy')
# volume = histogramdd(volume, bins=(128, 128, 128))[0]
# volume[:, :, 14:114] -= load('Phantoms/efg3cut.npy')*8
# volume = absolute(volume)

volume = loadtxt('Dat phantoms/efg3.dat').reshape((128, 128, 128), order='F')
print(unique(volume))

# data = rot90(data, k=1, axes=(2, 3))
# volume = data[0]
# volume = data.sum(axis=0)
mask = volume == 12.
# mask += volume == 0.1
volume[~mask] = 0
# volume[:, :, 60:] = 0

print(unique(volume))

size = array([
    128,
    128,
    128
])

# volume = []
# for x in open("Dat phantoms/efg3.dat"):
#     x = float(x)
#     # volume.append(x)
#     if x == 0.01: #Мягкие ткани
#         volume.append(0.019) #133
#     elif x == 0.015: #Лёгкие
#         volume.append(0.015) #107
#     elif x == 0.15: #печень
#         volume.append(0.15) #1020
#     elif x == 0.2: #толстая кишка
#         volume.append(0.3*100) 
#     elif x == 0.24: #сердце
#         volume.append(0.11) #735
#     elif x == 0.5: #желчный пузырь
#         volume.append(3.)
#     else:
#         volume.append(0.)
# volume_full = array(volume).reshape(size)

# volume_full = rot90(volume_full, k=1, axes=(0, 2))
# volume_full = rot90(volume_full, axes=(0, 1), k=1)
# volume_full = volume_full[::-1]

# volume = volume_full[59:85, 40:72, 51:74]
# radius = 1.2
# defect = zeros_like(volume)
# # defect = zoom(defect, radius, order=1)
# coordinates = array((18, 21, 11), dtype=int64)
# defect[coordinates[0], coordinates[1], coordinates[2]] = 1
# defect = gaussian_filter(defect, sigma=radius)
# coef = volume.max()/defect.max()
# # coef = 1
# # defect[defect > 0.02] = 0.15
# # defect[defect < 0.02] = 0
# mask = (volume == 0.15)
# volume -= defect*coef*mask
# volume = zoom(volume, 4, order=2)

# volume_full[59:85, 40:72, 51:74] = volume

# volume_full = volume_full[::-1]
# volume_full = rot90(volume_full, k=3, axes=(0, 1))
# volume_full = rot90(volume_full, k=3, axes=(0, 2))
# volume = volume_full

# volume_128 = zeros((128, 128, 128), float64)
# volume_128[14:114] += volume
# volume = volume_128

x0 = 13
y0 = 13
# volume = volume[:, y0:(y0 + 100), x0:(x0 + 100)]
# print(unique(volume))
# volume = zoom(volume, 0.64, order=1)
# print(unique(volume))

# volume = volume**(1/2)
voxel_size = 0.1
volume_size = array(volume.shape)*voxel_size
angles = linspace(0, 360, 32)

# with open(f'Dat phantoms/efg3_gaussDefect.dat', 'w') as file:
#     for value in volume.ravel():
#         file.write(str(value) + '\n')

levels = [
    min(volume),
    max(volume)
]

def gradientChanged():
    global volume, volumeItem, gradientEditor, levels, angles, horizontalSlider
    listTicks = gradientEditor.listTicks()
    listTicks[0][0].color.setAlpha(0)
    for tick in listTicks[1:]:
        tick[0].color.setAlpha(30)
    lut = gradientEditor.getLookupTable(255)
    volume_colors = asarray([makeRGBA(data=slice, lut=lut, levels=levels)[0] for slice in volume])
    # volume_colors[:, :, :, 3] = 50
    volumeItem.setData(volume_colors)

def sliderValueChanged():
    global volume, volumeViewWidget, angles, horizontalSlider
    volume = data[horizontalSlider.value()]
    gradientChanged()

def angleChanged():
    global volumeViewWidget, angles, horizontalSlider
    angle = angles[horizontalSlider.value()]
    volumeViewWidget.setCameraPosition(azimuth=angle)

app = mkQApp()
mainWindow = loadUi("UI/volume_visualization.ui")

volumeItem = GLVolumeItem(None, sliceDensity=10)
volumeItem.scale(*[voxel_size]*3)
gradientEditor = GradientEditorItem(orientation='right')
gradientEditor.sigGradientChangeFinished.connect(gradientChanged)
mainWindow.graphicsView.addItem(gradientEditor)
volumeViewWidget = mainWindow.openGLWidget
volumeViewWidget.addItem(volumeItem)
xs = GLAxisItem()
volumeViewWidget.addItem(xs)
xs.setSize(*volume_size*2)
horizontalSlider = mainWindow.horizontalSlider
horizontalSlider.valueChanged.connect(sliderValueChanged)

volumeViewWidget.setCameraPosition(distance=volume_size[0]*3, elevation=0)
volumeViewWidget.pan(*volume_size/2)

# collimator_box = GLBoxItem()
# collimator_box.setColor((0, 255, 0, 255))
# collimator_box.setSize(*collimator_size)
# collimator_box.rotate(collimator_angles[1]*180/pi, True, False, False)
# collimator_box.translate(*collimator_coordinates)
# mainWindow.openGLWidget.addItem(collimator_box)

# detector_box = GLBoxItem()
# detector_box.setColor((255, 0, 0, 255))
# detector_box.setSize(*detector_size)
# detector_box.rotate(detector_angles[1]*180/pi, True, False, False)
# detector_box.translate(*detector_coordinates)
# mainWindow.openGLWidget.addItem(detector_box)

space_box = GLBoxItem()
space_box.setColor((255, 255, 255, 255))
space_box.setSize(*volume_size)
mainWindow.openGLWidget.addItem(space_box)
gradientChanged()
mainWindow.show()
app.exec()