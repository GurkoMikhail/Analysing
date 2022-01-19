import numpy as np
from h5py import File
import SimpleITK as sitk


def save_as_dicom(filepath, data, voxel_size=0.4):
    data = (data*255/data.max()).astype(np.ubyte)
    shape = np.array(data.shape)
    data = data.ravel(order='F').reshape(shape[::-1])
    data = np.rot90(data, k=-1, axes=(0, 1))
    data = np.rot90(data, k=-1, axes=(0, 2))

    dicom_image = sitk.GetImageFromArray(data)
    dicom_image.SetSpacing([voxel_size*10]*3)
    dicom_image.SetMetaData('0010|0010', filepath.split('/')[-1])
    sitk.WriteImage(dicom_image, filepath + '.dcm')

def save_as_npy(filepath, data):
    np.save(filepath + '.npy', data)

def save_as_dat(filepath, data):
    data = data.ravel(order='F')
    np.savetxt(filepath + '_changed.dat', data, fmt='\t%.6E')

def from_dat_to_npy(filepath, size):
    data = np.loadtxt(filepath + '.dat').reshape(size, order='F')
    return data

def from_hdf_to_npy(filepath, distribution='Dose'):
    data = File(filepath + '.hdf')
    distribution = data[distribution + ' distribution']
    volume = np.array(distribution['Volume'])[::-1]
    voxel_size = np.array(distribution['Voxel size'])
    return volume, voxel_size

def change_values(data):

    data[data == 8] /= 5
    data[data == 10] /= 5
    data[data == 12] /= 5    

    # data[data == 0.] = 0    #Воздух
    # data[data == 0.04] = 1  #Лёгкие
    # data[data == 0.15] = 2   #Мягкие ткани
    # data[data == 0.28] = 3    #Кости


size = np.array([
    128,
    128,
    128
])
phantom_name = 'MMT'
filepathDAT = f'Dat phantoms/{phantom_name}'
filepathHDF = f'Raw data/{phantom_name}'
filepathNPY = f'Numpy phantoms/{phantom_name}'
filepathDICOM = f'DICOM phantoms/{phantom_name}'

data = from_dat_to_npy(filepathDAT, size)
print(np.unique(data))

# change_values(data)
# data = data.astype(np.uint)
# print(np.unique(data))

# data = data[:, :, 8:108]


# data = data[::-1]
# data = np.rot90(data, k=1, axes=(1, 2))
# data = np.rot90(data, k=-1, axes=(0, 2))

save_as_npy(filepathNPY, data)
save_as_dicom(filepathDICOM, data)

# save_as_dat(filepathDAT, data)

