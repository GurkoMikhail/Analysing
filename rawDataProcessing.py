import SimpleITK as sitk
from h5py import File
import numpy as np
from copy import deepcopy
from numba import jit


class DataSeries:

    light_speed = 2.99792458*10**10 # cm/s

    def __init__(self, nameList):
        self.nameList = nameList
        self.rngEnergy = np.random.default_rng()
        self.rngCoordinates = np.random.default_rng()
        self.fileDataList = {}
        self._readFiles()

    def _dataExtraction(self, file):
        data = {}
        if file['Modeling parameters/Subject'] is None:
            parameters = file['Modeling parameters/Space/Detector']
            # detector = Detector(
            #     coordinates=parameters['coordinates'],
            #     size=parameters['size'],
            #     euler_angles=parameters['euler_angles'],
            #     rotation_center=parameters['rotation_center']
            # )
        else:
            data = {
                'Detector size': None,
                'Interactions data': {}
            }
            interactions_data = file['Interactions data']
            for key in interactions_data.keys():
                data['Interactions data'].update({key: np.copy(interactions_data[key])})
            detector = file['Modeling parameters/Subject'].asstr()[()]
            data['Detector size'] = np.copy(file[f'Modeling parameters/Space/{detector}/size'])
            indicesSort = np.argsort(data['Interactions data']['Emission time'])
            interactions_data = data['Interactions data']
            for key in interactions_data.keys():
                interactions_data[key] = interactions_data[key][indicesSort]
        return data
            
    def convertToImage(self, data, imageParameters):
        print('converting to image')
        data = deepcopy(data)
        interactions_data = data['Interactions data']
        if imageParameters['imageRange'] is None:
            imageSize = data['Detector size'][:2]
            imageRange = np.column_stack([[0, 0], imageSize])
        else:
            imageRange = np.asarray(imageParameters['imageRange'])
        if imageParameters['matrix'] is None:
            matrix = ((imageRange[:, 1] - imageRange[:, 0])/imageParameters['pixelSize']).astype(int)
        else:
            matrix = np.asarray(imageParameters['matrix']).astype(int)
        self._emissionSlice(interactions_data, imageParameters['emissionSlice'])
        self._averageActs(interactions_data, imageParameters['decayTime'])
        self._energyDeviation(interactions_data, imageParameters['energyResolution'])
        self._energyWindow(interactions_data, imageParameters['energyWindow'])
        self._coordinatesDeviation(interactions_data, imageParameters['spatialResolution'])
        coordinates = interactions_data['Coordinates']
        imageArray = np.histogram2d(coordinates[:, 0], coordinates[:, 1], bins=matrix, range=imageRange)[0]
        return imageArray

    def acquireEnergySpectrum(self, data, spectumParameters):
        print('acquiring spectrum')
        data = deepcopy(data)
        interactions_data = data['Interactions data']
        self._emissionSlice(interactions_data, spectumParameters['emissionSlice'])
        self._averageActs(interactions_data, spectumParameters['decayTime'])
        self._energyDeviation(interactions_data, spectumParameters['energyResolution'])
        energySpectrum = np.histogram(interactions_data['Energy transfer'], bins=spectumParameters['energyChannels'], range=spectumParameters['energyRange'])
        return energySpectrum

    def energySpectrums(self, parameters={}):
        spectumParameters = {
            'decayTime': 300*10**(-9),
            'energyResolution': 9.9,
            'energyChannels': 1024,
            'energyRange': [0, 300*10**3],
            'emissionSlice': [0., 10.**10]
        }
        for key in spectumParameters.keys():
            if key in parameters:
                spectumParameters[key] = parameters[key]
        energySpectrums = []
        for data in self.fileDataList.values():
            energySpectrum = self.acquireEnergySpectrum(data, spectumParameters)
            energySpectrums.append(energySpectrum[0])
        energySpectrums.append(energySpectrum[1][1:] - (energySpectrum[1][1] - energySpectrum[1][0]))
        return np.stack(energySpectrums)

    def imagesArray(self, parameters={}):
        imageParameters ={
            'decayTime': 300*10**(-9),
            'spatialResolution': 0.4,
            'energyResolution': 9.9,
            'energyWindow': [126*10**3, 154*10**3],
            'imageRange': None,
            'pixelSize': 0.5,
            'matrix': None,
            'emissionSlice': [0., 10.**10]
        }
        for key in imageParameters.keys():
            if key in parameters:
                imageParameters[key] = parameters[key]
        images = []
        for data in self.fileDataList.values():
            image = self.convertToImage(data, imageParameters)
            images.append(image)
        return np.stack(images)

    @property
    def dataList(self):
        return self.fileDataList

    def _emissionSlice(self, data, emissionSlice):
        coordinates = data['Coordinates']
        energyTransfer = data['Energy transfer']
        emissionCoordinates = data['Emission coordinates']
        emissionTime = data['Emission time']
        indices = np.nonzero((emissionCoordinates[:, 1] >= emissionSlice[0])*(emissionCoordinates[:, 1] <=emissionSlice[1]))
        data['Coordinates'] = coordinates[indices]
        data['Emission time'] = emissionTime[indices]
        data['Energy transfer'] = energyTransfer[indices]
        data['Emission coordinates'] = emissionCoordinates[indices]

    def _energyWindow(self, data, energyWindow):
        coordinates = data['Coordinates']
        emissionTime = data['Emission time']
        energyTransfer = data['Energy transfer']
        emissionCoordinates = data['Emission coordinates']
        indices = np.nonzero((energyTransfer >= energyWindow[0])*(energyTransfer <= energyWindow[1]))
        data['Coordinates'] = coordinates[indices]
        data['Emission time'] = emissionTime[indices]
        data['Energy transfer'] = energyTransfer[indices]
        data['Emission coordinates'] = emissionCoordinates[indices]

    def _energyDeviation(self, data, energyResolution):
        energy = data['Energy transfer']
        coeff = np.sqrt(0.14)*energyResolution/100
        resolutionDistribution = coeff/np.sqrt(energy/10**6)
        sigma = resolutionDistribution*energy/2.355
        energy[:] = self.rngEnergy.normal(energy, sigma)

    def _coordinatesDeviation(self, data, spatialResolution):
        coordinates = data['Coordinates']
        sigma = spatialResolution/2.35
        coordinates[:] = self.rngCoordinates.normal(coordinates, sigma)

    def _averageActs(self, data, decayTime):
        coordinates = data['Coordinates']
        energyTransfer = data['Energy transfer']
        emissionCoordinates = data['Emission coordinates']
        emissionTime = data['Emission time']
        distanceTraveled = data['Distance traveled']
        registrationTime = emissionTime + distanceTraveled/self.light_speed
        timeWithDecay = self.withDecayTime(registrationTime, decayTime)
        unique, indices, counts = np.unique(timeWithDecay, return_index=True, return_counts=True)
        eventsNumber = indices.size
        eventsIndices = [np.arange(indices[i], indices[i] + counts[i]) for i in range(eventsNumber)]
        averagedCoordinates = np.zeros((eventsNumber, 3), dtype=float)
        averagedEmissionTime = np.zeros(eventsNumber, dtype=float)
        averagedEnergyTransfer = np.zeros(eventsNumber, dtype=float)
        averagedEmissionCoordinates = np.zeros((eventsNumber, 3), dtype=float)
        delIndices = []
        for i, actsIndices in enumerate(eventsIndices):
            weights = energyTransfer[actsIndices]
            if weights.sum() == 0.:
                delIndices.append(i)
                continue
            averagedCoordinates[i] = np.average(coordinates[actsIndices], axis=0, weights=weights)
            averagedEmissionTime[i] = emissionTime[actsIndices[0]]
            averagedEnergyTransfer[i] = np.sum(energyTransfer[actsIndices])
            averagedEmissionCoordinates[i] = emissionCoordinates[actsIndices[0]]
        delIndices = np.array(delIndices, dtype=int)
        data['Coordinates'] = np.delete(averagedCoordinates, delIndices, axis=0)
        data['Emission time'] = np.delete(averagedEmissionTime, delIndices, axis=0)
        data['Energy transfer'] = np.delete(averagedEnergyTransfer, delIndices, axis=0)
        data['Emission coordinates'] = np.delete(averagedEmissionCoordinates, delIndices, axis=0)

    def _readFiles(self):
        for name in self.nameList:
            print(f'reading {name}')
            file = File(f'Raw data/{name}', 'r')
            data = self._dataExtraction(file)
            self.fileDataList.update({name: data})
            file.close()

    @staticmethod
    @jit(nopython=True, cache=True)
    def withDecayTime(time, decayTime):
        timeWithDecay = np.zeros_like(time)
        countdownTime = 0.
        for i, t in enumerate(time):
            if (t - countdownTime) > decayTime:
                countdownTime = t
            timeWithDecay[i] = countdownTime + decayTime
        return timeWithDecay


def get_images(nameList):
    data = DataSeries(nameList)
    imageParameters ={
        'decayTime': 300*10**(-9),
        'spatialResolution': 0.4,
        'energyResolution': 9.9,
        'energyWindow': [126*10**3, 154*10**3],
        'pixelSize': 0.5,
        'energyChannels': 512*2,
        'energyRange': [0, 300*10**3],
        'emissionSlice': [0., 100.]
    }
    images = data.imagesArray(imageParameters)
    images = np.rot90(images, k=1, axes=(1, 2))
    return images


def save_as_dicom(phantom_name, data, pixel_size=0.5):
    data = (data*255/data.max()).astype(np.ubyte)
    dicom_image = sitk.GetImageFromArray(data)
    dicom_image.SetSpacing([pixel_size*10, pixel_size*10, 1])
    dicom_image.SetMetaData('0010|0010', phantom_name)
    sitk.WriteImage(dicom_image, f'DICOM data/{phantom_name}.dcm')


if __name__ == '__main__':
    phantom_name = 'efg3_full_angle'
    angles = np.linspace(-45., 90., 4)
    angles = [angles[1], ]
    nameList = [phantom_name + f' {angle}' +' deg' for angle in angles]
    phantom_name = 'point_source'
    nameList = ['point_source 10.0 cm', ]

    phantom_name = 'collimators'    
    nameList = [
        'SiemensSymbiaTSeriesLEHR',
        'SiemensSymbiaTSeriesLEAP',
        'SiemensSymbiaTSeriesLEUHR',
        'SiemensSymbiaTSeriesME',
        'SiemensSymbiaTSeriesHE'
        ]

    nameList = [name + '.hdf' for name in nameList]
    images = get_images(nameList)
    save_as_dicom(phantom_name, images)

