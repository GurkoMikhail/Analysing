from PyQt5 import QtGui
from PyQt5.uic import loadUi
from numpy import array, asarray, min, max, load, log, sqrt, linspace
from pyqtgraph.Qt import mkQApp
from pyqtgraph import GradientEditorItem, makeRGBA
from pyqtgraph.opengl import GLBoxItem, GLVolumeItem
from time import sleep
import h5py


# # file_name = 'efg3cut -1.5 deg.hdf'
# file_name = 'Attenuation test Water.hdf'
# # file_name = 'Attenuation inner test Bone.hdf'
# # file_name = 'efg3_fix front projection 5 sm.hdf'
# # file_name = 'efg3_fix front projection.hdf'
# # file_name = 'efg3_fix front projection 5 sm without collimator.hdf'

# file = h5py.File(f'Processed data/{file_name}')
# file = h5py.File(f'Raw data/{file_name}')
# dose_distribution = file['Dose distribution']
# volume = array(dose_distribution['Volume'])
# # volume = log(volume + 2)
# voxel_size = array(dose_distribution['Voxel size'])
# file.close()

volume = load('Phantoms/fgr_iter35_delta_0_03_gamm_0_01_V_Huber.npy')
voxel_size = 0.4
volume_size = array(volume.shape)*voxel_size
angles = linspace(-45, 135, 32)

levels = [
    min(volume),
    max(volume)
]

def gradientChanged():
    global volume, volumeItem, gradientEditor, levels, angles, horizontalSlider_angles
    listTicks = gradientEditor.listTicks()
    listTicks[0][0].color.setAlpha(0)
    for tick in listTicks[1:]:
        tick[0].color.setAlpha(50)
    lut = gradientEditor.getLookupTable(255)
    volume_colors = asarray([makeRGBA(data=slice, lut=lut, levels=levels)[0] for slice in volume])
    # volume_colors[:, :, :, 3] = 50
    volumeItem.setData(volume_colors)

def angleChanged():
    global volumeViewWidget, angles, horizontalSlider_angles
    angle = angles[horizontalSlider_angles.value()]
    volumeViewWidget.setCameraPosition(azimuth=angle)

mkQApp()
mainWindow = loadUi("UI/volume_visualization.ui")

volumeItem = GLVolumeItem(None, sliceDensity=2)
volumeItem.scale(*[voxel_size]*3)
gradientEditor = GradientEditorItem(orientation='right')
gradientEditor.sigGradientChangeFinished.connect(gradientChanged)
mainWindow.graphicsView.addItem(gradientEditor)
volumeViewWidget = mainWindow.openGLWidget
volumeViewWidget.addItem(volumeItem)
horizontalSlider_angles = mainWindow.horizontalSlider_angles
horizontalSlider_angles.valueChanged.connect(angleChanged)

volumeViewWidget.setCameraPosition(distance=volume_size[0]*3, elevation=0)
volumeViewWidget.pan(*volume_size/2)

space_box = GLBoxItem()
space_box.setSize(*volume_size)
mainWindow.openGLWidget.addItem(space_box)
gradientChanged()
mainWindow.show()
QtGui.QApplication.exec()