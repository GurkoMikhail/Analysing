from h5py import File
import numpy as np

acfile = File('tables/attenuationCoefficients.hdf', 'a')
materialsFile = File('tables/materials.hdf', 'r')
macGroup = acfile['NIST/Mass attenuation coefficients/Compounds and mixtures']
try:
    lacGroup = acfile['NIST/Linear attenuation coefficients/Compounds and mixtures']
except Exception:
    lacGroup = acfile.create_group('NIST/Linear attenuation coefficients/Compounds and mixtures')
materialsGroup = materialsFile['NIST/Compounds and mixtures']
for element, effects in macGroup.items():
    elementGroup = lacGroup.create_group(element)
    for effect, array in effects.items():
        if effect == 'Energy':
            elementGroup.create_dataset(effect, data=array)
            continue
        data = np.copy(array)*materialsGroup[f'{element}/Density']
        elementGroup.create_dataset(effect, data=data)
acfile.close()
materialsFile.close()

