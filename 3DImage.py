import numpy as np
from scipy.ndimage import zoom
from pyqtgraph.Qt import QtGui
import pyqtgraph as pg
import h5py

pg.setConfigOptions(imageAxisOrder='row-major')

app = QtGui.QApplication([])

win = QtGui.QMainWindow()
win.resize(1600, 900)
imv = pg.ImageView()
win.setCentralWidget(imv)
data = np.load('Processed data/collimators 10 cm.npy')

# file_name = 'collimators.hdf'
# file_name = 'Attenuation test Water 511.hdf'
# file_name = 'Attenuation inner test Bone.hdf'
# file_name = 'efg3_fix front projection 5 sm.hdf'
# file_name = 'efg3_fix front projection.hdf'
# file_name = 'efg3_fix front projection 5 sm without collimator.hdf'

# file = h5py.File(f'Processed data/{file_name}')
# file = h5py.File(f'Raw data/{file_name}')
# dose_distribution = file['Dose distribution']
# data = np.array(dose_distribution['Volume'])

# dat = []
# for slice in data:
#     print(slice.sum())
#     dat.append(zoom(slice, 10, order=1))
# data = np.asarray(dat)
# data = np.rot90(data, k=1, axes=(0, 2))
data = np.rot90(data, k=1, axes=(1, 2))
# data = zoom(data, 10, order=1)
imv.setImage(data)
win.show()
QtGui.QApplication.instance().exec_()
