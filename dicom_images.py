# from subjects import Detector
from pyqtgraph.graphicsItems.PlotCurveItem import PlotCurveItem
import utilites
# import pydicom
from h5py import File
import numpy as np
from copy import deepcopy

class DataSeries:

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
            coordinates = []
            energyTransfer = []
            emissionTime = []
            emissionCoordinates = []
            for flow in file['Flows'].values():
                coordinates.append(np.copy(flow['Coordinates']))
                energyTransfer.append(np.copy(flow['Energy transfer']))
                emissionTime.append(np.copy(flow['Emission time']))
                emissionCoordinates.append(np.copy(flow['Emission coordinates']))
            detectorSize = np.copy(file['Modeling parameters/Space/Detector/size'])
            emissionTime = np.concatenate(emissionTime)
            indicesSort = np.argsort(emissionTime)
            emissionTime = emissionTime[indicesSort]
            coordinates = np.concatenate(coordinates)[indicesSort]
            energyTransfer = np.concatenate(energyTransfer)[indicesSort]
            emissionCoordinates = np.concatenate(emissionCoordinates)[indicesSort]
            data.update({'Detector size': detectorSize})
            data.update({'Coordinates': coordinates})
            data.update({'Energy transfer': energyTransfer})
            data.update({'Emission time': emissionTime})
            data.update({'Emission coordinates': emissionCoordinates})
        return data
            
    def convertToImage(self, data, imageParameters):
        data = deepcopy(data)
        imageSize = data['Detector size'][:2]
        rangeSize = np.column_stack([[0, 0], imageSize])
        matrix = (imageSize/imageParameters['pixelSize']).astype(np.int64)
        self._emissionSlice(data, imageParameters['emissionSlice'])
        # self._averageActs(data, imageParameters['decayTime'])
        self._energyDeviation(data, imageParameters['energyResolution'])
        self._energyWindow(data, imageParameters['energyWindow'])
        self._coordinatesDeviation(data, imageParameters['spatialResolution'])
        coordinates = data['Coordinates']
        imageArray = np.histogram2d(coordinates[:, 0], coordinates[:, 1], bins=matrix, range=rangeSize)[0]
        return imageArray

    def acquireEnergySpectrum(self, data, spectumParameters):
        data = deepcopy(data)
        self._emissionSlice(data, spectumParameters['emissionSlice'])
        self._averageActs(data, spectumParameters['decayTime'])
        self._energyDeviation(data, spectumParameters['energyResolution'])
        energySpectrum = np.histogram(data['Energy transfer'], bins=spectumParameters['bins'], range=spectumParameters['energyRange'])
        return energySpectrum

    def energySpectrums(self, parameters={}):
        spectumParameters = {
            'decayTime': 300*10**(-9),
            'energyResolution': 9.9,
            'bins': 1000,
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
            'pixelSize': 0.6,
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
        emissionTime = data['Emission time']
        energyTransfer = data['Energy transfer']
        emissionCoordinates = data['Emission coordinates']
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
        emissionTime = data['Emission time']
        energyTransfer = data['Energy transfer']
        emissionCoordinates = data['Emission coordinates']
        timeWithDecay = utilites.withDecayTime(emissionTime, decayTime)
        unique, indices, counts = np.unique(timeWithDecay, return_index=True, return_counts=True)
        eventsNumber = indices.size
        eventsIndices = [np.arange(indices[i], indices[i] + counts[i]) for i in range(eventsNumber)]
        averagedCoordinates = np.zeros((eventsNumber, 3), dtype=np.float64)
        averagedEmissionTime = np.zeros(eventsNumber, dtype=np.float64)
        averagedEnergyTransfer = np.zeros(eventsNumber, dtype=np.float64)
        averagedEmissionCoordinates = np.zeros((eventsNumber, 3), dtype=np.float64)
        for i, actsIndices in enumerate(eventsIndices):
            weights = energyTransfer[actsIndices]
            averagedCoordinates[i] = np.average(coordinates[actsIndices], axis=0, weights=weights)
            averagedEmissionTime[i] = emissionTime[actsIndices[0]]
            averagedEnergyTransfer[i] = np.sum(energyTransfer[actsIndices])
            averagedEmissionCoordinates[i] = emissionCoordinates[actsIndices[0]]
        data['Coordinates'] = averagedCoordinates
        data['Emission time'] = averagedEmissionTime
        data['Energy transfer'] = averagedEnergyTransfer
        data['Emission coordinates'] = averagedEmissionCoordinates

    def _readFiles(self):
        for name in self.nameList:
            file = File(f'Raw data/{name}', 'r')
            data = self._dataExtraction(file)
            self.fileDataList.update({name: data})
            file.close()


def main():
    # angles = np.round(np.linspace(np.pi/4, -3*np.pi/4, 32)*180/np.pi, 1)
    # distances = np.round(np.linspace(2.5, 80.0, 32), 1)
    # angles = [1.5, -10.2]
    # nameList = ['efg3_fix ' + f'{angle}' +' deg.hdf' for angle in angles]
    # nameList = ['Point source ' + f'{distance}' +' sm.hdf' for distance in distances]
    nameList = ['LEHR', 'LEAP', 'LEUHR', 'ME']
    nameList = ['point source ' + name + ' 20.0 cm'+ '.hdf' for name in nameList]
    # nameList = [name + '.hdf' for name in nameList]
    data = DataSeries(nameList)
    imageParameters ={
        'decayTime': 0*10**(-9),
        'spatialResolution': 0.01,
        'energyResolution': 0.,
        'energyWindow': [140.5*10**3, 140.5*10**3],
        'pixelSize': 0.01,
        'emissionSlice': [0., 80.]
    }
    images = data.imagesArray(imageParameters)
    np.save('Processed data/collimators 20 cm.npy', images)
    energySpectrums = data.energySpectrums()
    np.save('Processed data/collimatorsEnergySpectrums 20 cm.npy', energySpectrums)

if __name__ == '__main__':
    main()
    exit()
    from pyqtgraph.graphicsItems.LinearRegionItem import LinearRegionItem
    from scipy import ndimage
    from pyqtgraph.Qt import QtGui
    import pyqtgraph as pg
    
    pg.setConfigOption('background', 'w')
    pg.setConfigOption('foreground', 'k')

    # angles = np.round(np.linspace(np.pi/4, -3*np.pi/4, 32)*180/np.pi, 1)
    # nameList = (['efg3cut ' + f'{angle}' +' deg.hdf' for angle in angles][8], )
    nameList = ['LEHR.hdf', 'LEAP.hdf', 'LEUHR.hdf', 'ME.hdf']
    data = DataSeries(nameList)
    detectorSize = data.dataList[nameList[1]]['Detector size']

    imageParameters ={
        'decayTime': 0*10**(-9),
        'spatialResolution': 0.01,
        'energyResolution': 0.,
        'energyWindow': [140.5*10**3, 140.5*10**3],
        'pixelSize': 0.01,
        'emissionSlice': [0., 70.]
    }

    pg.mkQApp()
    win = pg.GraphicsLayoutWidget()
    # win.setWindowTitle(file_name)

    pslices = win.addPlot()
    pslices.setLabel('left', "Y Axis", units='m')
    pslices.setLabel('bottom', "N")
    pslices.setMaximumWidth(150)
    win.nextColumn()
    emissionRegion = LinearRegionItem(np.array(imageParameters['emissionSlice'])/100, 'horizontal')
    
    def emissionSliceChanged():
        global imageParameters, data, emissionRegion, pslices
        sliceHistogramm = np.histogramdd(data.dataList[nameList[0]]['Emission coordinates'], bins=(1, 200, 1))
        imageParameters['emissionSlice'] = np.array(emissionRegion.getRegion())*100
        pslices.plot(y=sliceHistogramm[1][1][1:]/100, x=sliceHistogramm[0][0, :, 0], clear=True).setPen((0, 0, 0, 255))
        pslices.addItem(emissionRegion)
        energySpectrumChanged()
        updateImage()

    emissionRegion.sigRegionChangeFinished.connect(emissionSliceChanged)

    p1 = win.addPlot()
    # p1.setTitle(f'matrix = {matrix}, pixel size = {round(pixel_size*10, 2)} mm, resolution = {round(resolution*10, 2)} mm')
    p1.setLabel('left', "Z Axis", units='m')
    p1.setLabel('bottom', "X Axis", units='m')
    p1.getViewBox().setAspectLocked(detectorSize[0]/detectorSize[1])

    image = pg.ImageItem()
    p1.addItem(image)
    p1.getAxis('left').setZValue(image.zValue() + 1)
    p1.getAxis('bottom').setZValue(image.zValue() + 1)
    image.scale(imageParameters['pixelSize']/100, imageParameters['pixelSize']/100)
    # image.translate(*(-detector_size[:2]/100/2))
    # p1.showGrid(True, True)

    hist = pg.HistogramLUTItem()
    hist.setMaximumWidth(150)
    hist.setImageItem(image)
    win.addItem(hist)

    win.nextRow()
    p2 = win.addPlot(colspan=4, title='Energy spectrum')
    p2.setLabel('left', 'N')
    p2.setLabel('bottom', 'Energy', units='eV')
    p2.setLogMode(y=True)
    p2.showGrid(x=True, y=True)
    p2.setMaximumHeight(250)
    win.resize(1200, 1000)
    energy_region = LinearRegionItem(imageParameters['energyWindow'])

    def energySpectrumChanged():
        global energy_region, data
        energySpectrum = data.energySpectrums(imageParameters)
        energySpectrum += 1
        p2.plot(x=energySpectrum[1], y=energySpectrum[0], clear=True).setPen((0, 0, 0, 255))
        p2.addItem(energy_region)

    def energyWindowChanged():
        global energy_region, imageParameters
        imageParameters['energyWindow'] = energy_region.getRegion()
        updateImage()

    energy_region.sigRegionChangeFinished.connect(energyWindowChanged)

    def updateImage():
        global image, data, imageParameters
        image.setImage(data.imagesArray(imageParameters)[0])

    emissionSliceChanged()

    win.show()

    QtGui.QApplication.instance().exec_()


