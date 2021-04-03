from PyQt5 import QtGui
from PyQt5.uic import loadUi
from numpy import array, asarray, min, max, load, log, sqrt
from pyqtgraph.Qt import mkQApp
from pyqtgraph import GradientEditorItem, makeRGBA
from pyqtgraph.opengl import GLBoxItem, GLVolumeItem
import h5py


# file_name = 'efg3cut -1.5 deg.hdf'
file_name = 'Attenuation test Water.hdf'
# file_name = 'Attenuation inner test Bone.hdf'
# file_name = 'efg3_fix front projection 5 sm.hdf'
# file_name = 'efg3_fix front projection.hdf'
# file_name = 'efg3_fix front projection 5 sm without collimator.hdf'

# file = h5py.File(f'Processed data/{file_name}')
file = h5py.File(f'Raw data/{file_name}')
dose_distribution = file['Dose distribution']
volume = array(dose_distribution['Volume'])
volume = (volume)**(1/2)
voxel_size = array(dose_distribution['Voxel size'])
file.close()
# volume = load('Phantoms/efg3cut.npy')
# voxel_size = 0.4
volume_size = array(volume.shape)*voxel_size

levels = [
    min(volume),
    max(volume)
]

def gradientChanged():
    global volume, volumeItem, gradientEditor, levels
    listTicks = gradientEditor.listTicks()
    listTicks[0][0].color.setAlpha(0)
    for tick in listTicks[1:]:
        tick[0].color.setAlpha(30)
    lut = gradientEditor.getLookupTable(255)
    volume_colors = asarray([makeRGBA(data=slice, lut=lut, levels=levels)[0] for slice in volume])
    volumeItem.setData(volume_colors)

mkQApp()
mainWindow = loadUi("UI/volume_visualization.ui")

volumeItem = GLVolumeItem(None, sliceDensity=2)
volumeItem.scale(*[voxel_size]*3)
gradientEditor = GradientEditorItem(orientation='right')
gradientEditor.sigGradientChangeFinished.connect(gradientChanged)
mainWindow.graphicsView.addItem(gradientEditor)
volumeViewWidget = mainWindow.openGLWidget
volumeViewWidget.addItem(volumeItem)

volumeViewWidget.setCameraPosition(distance=volume_size[0]*3)
volumeViewWidget.pan(*volume_size/2)

space_box = GLBoxItem()
space_box.setSize(*volume_size)
mainWindow.openGLWidget.addItem(space_box)
gradientChanged()
mainWindow.show()
QtGui.QApplication.exec()