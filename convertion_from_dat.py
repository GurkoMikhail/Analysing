import numpy as np
from pyevtk.hl import gridToVTK
import SimpleITK as sitk


def save_as_vtk(filepath, data, voxel_size=0.4):
    x = np.arange(data.shape[0] + 1)*voxel_size*10
    y = np.arange(data.shape[1] + 1)*voxel_size*10
    z = np.arange(data.shape[2] + 1)*voxel_size*10
    gridToVTK(filepath, x, y, z, cellData={'data':data.copy()})

def save_as_dicom(filepath, data, voxel_size=0.4):
    data = (data*255/data.max()).astype(np.ubyte)
    dicom_image = sitk.GetImageFromArray(data)
    dicom_image.SetSpacing([voxel_size*10]*3)
    dicom_image.SetMetaData('0010|0010', filepath.split('/')[-1])
    sitk.WriteImage(dicom_image, filepath + '.dcm')

def from_dat_to_npy(filepath, size):
    data = [float(x) for x in open(filepath + '.dat')]
    data = np.asfarray(data).reshape(size, order='C')
    data = np.rot90(data, k=1, axes=(1, 2))
    return data


size = np.array([
    128,
    128,
    128
])
phantom_name = 'ae3'
filepathDAT = f'Dat phantoms/{phantom_name}'
filepathNPY = f'Numpy phantoms/{phantom_name}'
filepathVTK = f'VTK phantoms/{phantom_name}'
filepathDICOM = f'DICOM phantoms/{phantom_name}'

data = from_dat_to_npy(filepathDAT, size)
print(np.unique(data))
np.save(filepathNPY + '.npy', data)
save_as_dicom(filepathDICOM, data)
save_as_vtk(filepathVTK, data)

