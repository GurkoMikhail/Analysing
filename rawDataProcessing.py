import SimpleITK as sitk
from h5py import File
import numpy as np
from copy import deepcopy
from numba import jit
from multiprocessing import Pool


class DataExtractor:

    def __init__(self, maxProcesses=32):
        self.maxProcesses = maxProcesses

    def extractData(self, filesPath):
        if isinstance(filesPath, list):
            with Pool(self.maxProcesses) as pool:
                return pool.map(self._extractData, filesPath)
        return self._extractData(filesPath)

    def _extractData(self, filePath):
        fileName = filePath.split(sep="/")[-1]
        print(f'Reading {fileName}')
        file = File(filePath, 'r')
        data = self._dataExtraction(file)
        file.close()
        return data

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
            

class DataProcessor:

    light_speed = 2.99792458*10**10 # cm/s

    def __init__(self):
        self.rngEnergy = np.random.default_rng()
        self.rngCoordinates = np.random.default_rng()

    def addEmissionSlice(self, data, emissionSlice):
        coordinates = data['Coordinates']
        energyTransfer = data['Energy transfer']
        emissionCoordinates = data['Emission coordinates']
        emissionTime = data['Emission time']
        indices = np.nonzero((emissionCoordinates[:, 1] >= emissionSlice[0])*(emissionCoordinates[:, 1] <=emissionSlice[1]))
        data['Coordinates'] = coordinates[indices]
        data['Emission time'] = emissionTime[indices]
        data['Energy transfer'] = energyTransfer[indices]
        data['Emission coordinates'] = emissionCoordinates[indices]

    def addEnergyWindow(self, data, energyWindow):
        coordinates = data['Coordinates']
        emissionTime = data['Emission time']
        energyTransfer = data['Energy transfer']
        emissionCoordinates = data['Emission coordinates']
        indices = np.nonzero((energyTransfer >= energyWindow[0])*(energyTransfer <= energyWindow[1]))
        data['Coordinates'] = coordinates[indices]
        data['Emission time'] = emissionTime[indices]
        data['Energy transfer'] = energyTransfer[indices]
        data['Emission coordinates'] = emissionCoordinates[indices]

    def addEnergyDeviation(self, data, energyResolution):
        energy = data['Energy transfer']
        coeff = np.sqrt(0.14)*energyResolution/100
        resolutionDistribution = coeff/np.sqrt(energy/10**6)
        sigma = resolutionDistribution*energy/2.355
        energy[:] = self.rngEnergy.normal(energy, sigma)

    def addCoordinatesDeviation(self, data, spatialResolution):
        coordinates = data['Coordinates']
        sigma = spatialResolution/2.35
        coordinates[:] = self.rngCoordinates.normal(coordinates, sigma)

    def averageActs(self, data, decayTime, useDistanceTraveled=False):
        coordinates = data['Coordinates']
        energyTransfer = data['Energy transfer']
        emissionCoordinates = data['Emission coordinates']
        emissionTime = data['Emission time']
        distanceTraveled = data['Distance traveled']
        if useDistanceTraveled:
            registrationTime = emissionTime + distanceTraveled/self.light_speed
        else:
            registrationTime = emissionTime
        timeWithDecay = self.addDecayTime(registrationTime, decayTime)
        _, indices, counts = np.unique(timeWithDecay, return_index=True, return_counts=True)
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

    @staticmethod
    @jit(nopython=True, cache=True)
    def addDecayTime(time, decayTime):
        timeWithDecay = np.zeros_like(time)
        countdownTime = 0.
        for i, t in enumerate(time):
            if (t - countdownTime) > decayTime:
                countdownTime = t
            timeWithDecay[i] = countdownTime + decayTime
        return timeWithDecay
        

class DataConverter:

    def __init__(self, maxProcesses=32):
        self.maxProcesses = maxProcesses
        self.spectumParameters = {
            'decayTime': 300*10**(-9),
            'energyResolution': 9.9,
            'energyChannels': 1024,
            'energyRange': [0, 300*10**3],
            'useDistanceTraveled': True,
            'emissionSlice': [0., 10.**10]
        }
        self.imageParameters = {
            'decayTime': 300*10**(-9),
            'spatialResolution': 0.4,
            'energyResolution': 9.9,
            'energyWindow': [126*10**3, 154*10**3],
            'imageRange': [[2.9, 56.2], [2.9, 41.6]],
            'pixelSize': 0.6,
            'matrix': None,
            'useDistanceTraveled': True,
            'emissionSlice': [0., 10.**10]
        }
        self.scattersImageParameters = {
            'decayTime': 0.,
            'spatialResolution': 0.,
            'energyResolution': 0.,
            'energyWindow': [126*10**3, 154*10**3],
            'imageRange': [[0.29, 5.62], [0.29, 4.16]],
            'pixelSize': 0.6,
            'matrix': None,
            'useDistanceTraveled': False,
            'emissionSlice': [0., 10.**10],
            'peakEnergy': 140500
        }

    @staticmethod
    def updateParameters(parameters, newParameters):
        for key, value in newParameters.items():
            if key in parameters:
                parameters[key] = value

    def _getMatrixAndImageRange(self, data, parameters):
        if parameters['imageRange'] is None:
            imageSize = data['Detector size'][:2]
            imageRange = np.column_stack([[0, 0], imageSize])
        else:
            imageRange = np.asarray(parameters['imageRange'])
        if parameters['matrix'] is None:
            matrix = ((imageRange[:, 1] - imageRange[:, 0])/parameters['pixelSize']).astype(int)
        else:
            matrix = np.asarray(parameters['matrix']).astype(int)
        return matrix, imageRange

    def acquireEnergySpectrum(self, data, spectumParameters={}):
        print('\tAcquiring spectrum')
        self.updateParameters(self.spectumParameters, spectumParameters)
        if isinstance(data, list):
            with Pool(self.maxProcesses) as pool:
                return pool.map(self._acquireEnergySpectrum, data)
        return self._acquireEnergySpectrum(data)

    def convertToImage(self, data, imageParameters={}):
        print('\tConverting to image')
        self.updateParameters(self.imageParameters, imageParameters)
        if isinstance(data, list):
            with Pool(self.maxProcesses) as pool:
                return pool.map(self._convertToImage, data)
        return self._convertToImage(data)

    def convertToScattersImage(self, data, scattersImageParameters={}):
        print('\tConverting to scatters image')
        self.updateParameters(self.scattersImageParameters, scattersImageParameters)
        if isinstance(data, list):
            with Pool(self.maxProcesses) as pool:
                return pool.map(self._convertToScattersImage, data)
        return self._convertToScattersImage(data)

    def _acquireEnergySpectrum(self, data):
        dataProcessor = DataProcessor()
        data = deepcopy(data)
        spectumParameters = self.spectumParameters
        interactions_data = data['Interactions data']
        dataProcessor.addEmissionSlice(interactions_data, spectumParameters['emissionSlice'])
        dataProcessor.averageActs(interactions_data, spectumParameters['decayTime'], spectumParameters['useDistanceTraveled'])
        dataProcessor.addEnergyDeviation(interactions_data, spectumParameters['energyResolution'])
        energySpectrum = list(np.histogram(interactions_data['Energy transfer'], bins=spectumParameters['energyChannels'], range=spectumParameters['energyRange']))
        energySpectrum[1] = energySpectrum[1][1:]
        return energySpectrum

    def _convertToImage(self, data):
        dataProcessor = DataProcessor()
        data = deepcopy(data)
        imageParameters = self.imageParameters
        interactions_data = data['Interactions data']
        dataProcessor.addEmissionSlice(interactions_data, imageParameters['emissionSlice'])
        dataProcessor.averageActs(interactions_data, imageParameters['decayTime'], imageParameters['useDistanceTraveled'])
        dataProcessor.addEnergyDeviation(interactions_data, imageParameters['energyResolution'])
        dataProcessor.addEnergyWindow(interactions_data, imageParameters['energyWindow'])
        dataProcessor.addCoordinatesDeviation(interactions_data, imageParameters['spatialResolution'])
        coordinates = interactions_data['Coordinates']
        matrix, imageRange = self._getMatrixAndImageRange(data, imageParameters)
        imageArray = np.histogram2d(coordinates[:, 0], coordinates[:, 1], bins=matrix, range=imageRange)[0]
        return imageArray

    def _convertToScattersImage(self, data):
        dataProcessor = DataProcessor()
        data = deepcopy(data)
        scattersImageParameters = self.scattersImageParameters
        interactions_data = data['Interactions data']
        matrix, imageRange = self._getMatrixAndImageRange(data, scattersImageParameters)
        dataProcessor.addEmissionSlice(interactions_data, scattersImageParameters['emissionSlice'])
        dataProcessor.averageActs(interactions_data, scattersImageParameters['decayTime'], scattersImageParameters['useDistanceTraveled'])
        dataProcessor.addEnergyWindow(interactions_data, scattersImageParameters['energyWindow'])
        coordinates = interactions_data['Coordinates']
        indicesOfPeak = (interactions_data['Energy transfer'] == scattersImageParameters['peakEnergy']).nonzero()[0]
        peakImageArray = np.histogram2d(coordinates[indicesOfPeak, 0], coordinates[indicesOfPeak, 1], bins=matrix, range=imageRange)[0]
        generalImageArray = np.histogram2d(coordinates[:, 0], coordinates[:, 1], bins=matrix, range=imageRange)[0]
        nulls = generalImageArray == 0
        peakImageArray[nulls] = 1
        generalImageArray[nulls] = 1
        scattersImageArray = 1 - peakImageArray/generalImageArray
        return scattersImageArray


class DataSaver:

    def __init__(self, data, fileName, dataType=None, pixelSize=0.6):
        self.data = np.asarray(data)
        self._fileName = fileName
        self.dataType = dataType
        self.pixelSize = pixelSize

    @property
    def fileName(self):
        if self.data.ndim > 2:
            return self._fileName.split('/')[0] + ('' if self.dataType is None else '_' + self.dataType)
        return self._fileName.split('/')[-1] + ('' if self.dataType is None else '_' + self.dataType)

    def saveAsNumpy(self):
        print(f'Saving {self.fileName} as Numpy')
        np.save(f'Numpy data/{self.fileName}.npy', self.data)

    def saveAsDicom(self):
        print(f'Saving {self.fileName} as Dicom')
        if self.data.ndim > 2:
            data = np.rot90(self.data, axes=(1, 2))
            data = data[:, ::-1]
        else:
            data = np.rot90(self.data)
            data = data[::-1]
        data = data/self.data.max()*255
        data = data.astype(np.ubyte)
        image = sitk.GetImageFromArray(data, isVector=False)
        image.SetOrigin((0., 0., 0.))
        image.SetSpacing([self.pixelSize*10, self.pixelSize*10, 1])
        image.SetMetaData('0010|0010', self.fileName)
        sitk.WriteImage(image, f'DICOM data/{self.fileName}.dcm')

    def saveAsDat(self):
        print(f'Saving {self.fileName} as Dat')
        data = np.asfortranarray(self.data)
        if self.data.ndim > 2:
            from pathlib import Path
            Path(f'Dat data/{self.fileName}').mkdir(parents=True, exist_ok=True)
            for i, image in enumerate(data, 1):
                image = np.rot90(image)
                np.savetxt(f'Dat data/{self.fileName}/{i}.dat', image)
        else:
            data = np.rot90(data)
            np.savetxt(f'Dat data/{self.fileName}.dat', data)


if __name__ == '__main__':
    fileName = 'efg3_32'
    angles = np.linspace(45., -135., 32).round(1)
    nameList = [f'Raw data/{fileName}/{angle} deg' for angle in angles]
    nameList = [name + '.hdf' for name in nameList]

    dataExtractor = DataExtractor()
    data = dataExtractor.extractData(nameList)

    dataConverter = DataConverter()

    images = dataConverter.convertToImage(data)
    dataSaver = DataSaver(images, fileName)
    dataSaver.saveAsNumpy()
    dataSaver.saveAsDicom()
    dataSaver.saveAsDat()

    scattersImages = dataConverter.convertToScattersImage(data)
    dataSaver = DataSaver(scattersImages, fileName, 'scatters')
    dataSaver.saveAsNumpy()
    dataSaver.saveAsDicom()
    dataSaver.saveAsDat()

    energySpectrums = dataConverter.acquireEnergySpectrum(data)
    dataSaver = DataSaver(energySpectrums, fileName, 'spectrum')
    dataSaver.saveAsNumpy()

