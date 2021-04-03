from h5py import File
import numpy as np
angles = np.linspace(-np.pi/4, 3*np.pi/4, 32)

for angle in angles:
    file = File(f'Output data/efg3_fix {round(angle/np.pi*180, 1)} deg.hdf', 'a')
    group = file['Flows']
    del group['Coordinates']
    del group['Emission coordinates']
    del group['Emission time']
    del group['Energy transfer']
    file.close()