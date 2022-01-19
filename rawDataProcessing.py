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
            data['R'] = np.copy(file[f'Modeling parameters/Space/ae3/R'])
            data['Rotation center'] = np.copy(file[f'Modeling parameters/Space/ae3/rotation_center'])
            data['Origin'] = np.copy(file[f'Modeling parameters/Space/ae3/coordinates'])
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

    def addEmissionROI(self, data, emissionROI):
        coordinates = data['Coordinates']
        energyTransfer = data['Energy transfer']
        distanceTraveled = data['Distance traveled']
        emissionCoordinates = data['Emission coordinates']
        emissionTime = data['Emission time']
        emissionROI = np.asarray(emissionROI)
        ROI = np.all((emissionCoordinates >= emissionROI[:, 0])*(emissionCoordinates < emissionROI[:, 1]), axis=1)
        indices = np.nonzero(ROI)[0]
        data['Coordinates'] = coordinates[indices]
        data['Distance traveled'] = distanceTraveled[indices]
        data['Emission time'] = emissionTime[indices]
        data['Energy transfer'] = energyTransfer[indices]
        data['Emission coordinates'] = emissionCoordinates[indices]

    def addEnergyWindow(self, data, energyWindow):
        coordinates = data['Coordinates']
        distanceTraveled = data['Distance traveled']
        emissionTime = data['Emission time']
        energyTransfer = data['Energy transfer']
        emissionCoordinates = data['Emission coordinates']
        indices = np.nonzero((energyTransfer >= energyWindow[0])*(energyTransfer <= energyWindow[1]))
        data['Coordinates'] = coordinates[indices]
        data['Distance traveled'] = distanceTraveled[indices]
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
        averagedDistanceTraveled = np.zeros(eventsNumber, dtype=float)
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
            averagedDistanceTraveled[i] = np.average(distanceTraveled[actsIndices], weights=weights)
            averagedEmissionTime[i] = np.average(emissionTime[actsIndices], weights=weights)
            averagedEnergyTransfer[i] = np.sum(energyTransfer[actsIndices])
            averagedEmissionCoordinates[i] = np.average(emissionCoordinates[actsIndices], axis=0, weights=weights)
        delIndices = np.array(delIndices, dtype=int)
        data['Coordinates'] = np.delete(averagedCoordinates, delIndices, axis=0)
        data['Distance traveled'] = np.delete(averagedDistanceTraveled, delIndices)
        data['Emission time'] = np.delete(averagedEmissionTime, delIndices)
        data['Energy transfer'] = np.delete(averagedEnergyTransfer, delIndices)
        data['Emission coordinates'] = np.delete(averagedEmissionCoordinates, delIndices, axis=0)

    @staticmethod
    def convertToLocalCoordinates(coordinates, origin, R, rotationCenter):
        """ Преобразовать в локальные координаты """
        coordinates -= origin
        coordinates -= rotationCenter
        np.matmul(coordinates, R, out=coordinates)
        coordinates += rotationCenter
        coordinates += origin

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
        self.processingParameters = {
            'decayTime': 300*10**(-9),
            'spatialResolution': 0.4,
            'energyResolution': 9.9,
            'energyChannels': 1024,
            'energyRange': [0, 300*10**3],
            'energyWindow': [126*10**3, 154*10**3],
            'imageRange': [[0., 51.2], [0., 51.2]],
            'pixelSize': 0.4,
            'matrix': None,
            'useDistanceTraveled': True,
            'returnEnergySpectum': True,
            'returnEmissionDistribution': True,
            'emissionROI': [[0., 51.2], [0., 51.2], [3.855, 3.855 + 51.2]]
        }
        self.scattersImageParameters = {
            'decayTime': 0.,
            'spatialResolution': 0.,
            'energyResolution': 0.,
            'energyWindow': [126*10**3, 154*10**3],
            'imageRange': [[0., 51.2], [0., 51.2]],
            'pixelSize': 0.4,
            'matrix': None,
            'useDistanceTraveled': False,
            'emissionROI': [[0., 51.2], [0., 51.2], [3.855, 3.855 + 51.2]],
            'peakEnergy': 140500
        }

    def _getMatrixAndImageRange(self, data, parameters):
        if parameters['imageRange'] is None:
            imageSize = data['Detector size'][:2]
            imageRange = np.column_stack([[0, 0], imageSize])
        else:
            imageRange = np.asarray(parameters['imageRange'])
        if parameters['matrix'] is None:
            matrix = np.round(((imageRange[:, 1] - imageRange[:, 0])/parameters['pixelSize'])).astype(int)
        else:
            matrix = np.asarray(parameters['matrix']).astype(int)
        return matrix, imageRange

    def convertToImage(self, data, processingParameters={}):
        print('\tConverting to image')
        self.updateParameters(self.processingParameters, processingParameters)
        if isinstance(data, list):
            with Pool(min(len(data), self.maxProcesses)) as pool:
                return pool.map(self._convertToImage, data)
        return self._convertToImage(data)

    def convertToScattersImage(self, data, scattersImageParameters={}):
        print('\tConverting to scatters image')
        self.updateParameters(self.scattersImageParameters, scattersImageParameters)
        if isinstance(data, list):
            with Pool(min(len(data), self.maxProcesses)) as pool:
                return pool.map(self._convertToScattersImage, data)
        return self._convertToScattersImage(data)

    def _acquireEnergySpectrum(self, processedData):
        processingParameters = self.processingParameters
        energySpectrum = list(np.histogram(
            processedData['Energy transfer'],
            bins=processingParameters['energyChannels'],
            range=processingParameters['energyRange']
            ))
        energySpectrum[1] = energySpectrum[1][1:]
        return np.array(energySpectrum)

    def _obtainEmissionDistrubution(self, emissionCoordinates):
        processingParameters = self.processingParameters
        emissionRange = np.asarray(processingParameters['emissionROI'])
        bins = np.round((emissionRange[:, 1] - emissionRange[:, 0])/processingParameters['pixelSize']).astype(int)
        emissionDistribution = np.histogramdd(
            emissionCoordinates,
            bins=bins,
            range=emissionRange
            )
        return emissionDistribution[0]

    def _convertToImage(self, data):
        processedData = self.processData(data['Interactions data'], self.processingParameters)
        matrix, imageRange = self._getMatrixAndImageRange(data, self.processingParameters)
        validEvents = self.cutToEnergyWindow(processedData['Energy transfer'], self.processingParameters['energyWindow'])
        coordinates = processedData['Coordinates'][validEvents]
        imageArray = np.histogram2d(coordinates[:, 0], coordinates[:, 1], bins=matrix, range=imageRange)[0]
        if self.processingParameters['returnEnergySpectum'] or self.processingParameters['returnEmissionDistribution']:
            output = [imageArray]
            if self.processingParameters['returnEnergySpectum']:
                energySpectrum = self._acquireEnergySpectrum(processedData)
                output.append(energySpectrum)
            if self.processingParameters['returnEmissionDistribution']:
                emissionCoordinates = processedData['Emission coordinates'][validEvents]
                DataProcessor.convertToLocalCoordinates(
                    emissionCoordinates,
                    data['Origin'],
                    data['R'],
                    data['Rotation center']
                    )
                emissionDistrubution = self._obtainEmissionDistrubution(emissionCoordinates)
                output.append(emissionDistrubution)
            return output
        return imageArray

    def _convertToScattersImage(self, data):
        processedData = self.processData(data['Interactions data'], self.scattersImageParameters)
        matrix, imageRange = self._getMatrixAndImageRange(data, self.scattersImageParameters)
        coordinates = processedData['Coordinates']
        indicesOfPeak = (processedData['Energy transfer'] == self.scattersImageParameters['peakEnergy']).nonzero()[0]
        peakImageArray = np.histogram2d(coordinates[indicesOfPeak, 0], coordinates[indicesOfPeak, 1], bins=matrix, range=imageRange)[0]
        validEvents = self.cutToEnergyWindow(processedData['Energy transfer'], self.processingParameters['energyWindow'])
        generalImageArray = np.histogram2d(coordinates[validEvents, 0], coordinates[validEvents, 1], bins=matrix, range=imageRange)[0]
        nulls = generalImageArray == 0
        peakImageArray[nulls] = 1
        generalImageArray[nulls] = 1
        scattersImageArray = 1 - peakImageArray/generalImageArray
        return scattersImageArray

    @staticmethod
    def updateParameters(parameters, newParameters):
        for key, value in newParameters.items():
            if key in parameters:
                parameters[key] = value

    @staticmethod
    def cutToEnergyWindow(energyTransfer, energyWindow):
        return np.nonzero((energyTransfer >= energyWindow[0])*(energyTransfer <= energyWindow[1]))[0]

    @staticmethod
    def processData(interactions_data, processingParameters):
        dataProcessor = DataProcessor()
        interactions_data = deepcopy(interactions_data)
        dataProcessor.addEmissionROI(interactions_data, processingParameters['emissionROI'])
        dataProcessor.averageActs(interactions_data, processingParameters['decayTime'], processingParameters['useDistanceTraveled'])
        dataProcessor.addEnergyDeviation(interactions_data, processingParameters['energyResolution'])
        dataProcessor.addCoordinatesDeviation(interactions_data, processingParameters['spatialResolution'])
        return interactions_data


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

    def saveAsNumpy(self, rot=False):
        print(f'Saving {self.fileName} as Numpy')
        data = self.data
        if rot:
            if self.data.ndim > 2:
                data = np.rot90(self.data, k=-1, axes=(1, 2))
            else:
                data = np.rot90(self.data, k=-1)
        np.save(f'Numpy data/{self.fileName}.npy', data)

    def saveAsDicom(self):
        print(f'Saving {self.fileName} as Dicom')
        if self.data.ndim > 2:
            data = np.rot90(self.data, k=-1, axes=(1, 2))
            data = data[:, ::-1]
        else:
            data = np.rot90(self.data, k=-1)
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
        data = self.data
        if self.data.ndim > 2:
            from pathlib import Path
            Path(f'Dat data/{self.fileName}').mkdir(parents=True, exist_ok=True)
            for i, image in enumerate(data, 1):
                image = image[::-1]
                np.savetxt(f'Dat data/{self.fileName}/{i}.dat', image, fmt='%i', delimiter='\t')
        else:
            data = data[::-1]
            np.savetxt(f'Dat data/{self.fileName}.dat', data, fmt='%i', delimiter='\t')


if __name__ == '__main__':
    fileName = 'efg3_32'
    angles = np.linspace(-np.pi/4, 3*np.pi/4, 32)
    angles = np.round(np.rad2deg(angles), 1)
    nameList = [f'Raw data/{fileName}/{angle} deg' for angle in angles]
    nameList = [name + '.hdf' for name in nameList]

    dataExtractor = DataExtractor()
    data = dataExtractor.extractData(nameList)

    dataConverter = DataConverter()

    output = dataConverter.convertToImage(data)
    images = [image for image, spectrum, distribution in output]
    energySpectrums = [spectrum for image, spectrum, distribution in output]
    distributions = np.asarray([distribution for image, spectrum, distribution in output])

    dataSaver = DataSaver(images, fileName)
    dataSaver.saveAsNumpy(rot=True)
    dataSaver.saveAsDicom()
    dataSaver.saveAsDat()
    
    dataSaver = DataSaver(energySpectrums, fileName, 'spectrums')
    dataSaver.saveAsNumpy()

    dataSaver = DataSaver(distributions, fileName, 'distributions')
    dataSaver.saveAsNumpy()

    scattersImages = dataConverter.convertToScattersImage(data)
    dataSaver = DataSaver(scattersImages, fileName, 'scatters')
    dataSaver.saveAsNumpy()
    dataSaver.saveAsDicom()
    dataSaver.saveAsDat()

