from pyqtgraph.Qt import QtGui
import numpy as np
import pyqtgraph as pg

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

energySpectrums = np.load('Numpy data/efg3_32_spectrums.npy')
# energySpectrums = np.load('Processed data/PS10EnergySpectums.npy') + 1

pg.mkQApp()
win = pg.GraphicsLayoutWidget()
# win.resize(800, 400)
win.resize(1366, 768)

plots = []
i, j = 0, 0
for n, energySpectrum in enumerate(energySpectrums, 1):
    counts, energy = energySpectrum
    p = win.addPlot(col=j, row=i)
    # p.setTitle(f'Energy spectrum №{n}')
    # p.setLogMode(y=True)
    p.setLabel('left', 'Counts')
    p.setLabel('bottom', f'№{n} Energy', units='eV')
    p.showGrid(x=True, y=True)
    p.plot(x=energy, y=counts).setPen((0, 0, 255, 255))
    plots.append(p)
    if i < 8 - 1:
        i += 1
    else:
        j += 1
        i = 0

# p = win.addPlot(title='Energy spectrum')
# p.plot(x=energySpectrums[0, 1],y=(np.sum(energySpectrums[:, 0], axis=0) + 1)).setPen((0, 0, 255, 255))
# # p.setLogMode(y=True)
# p.showGrid(x=True, y=True)
# p.setLabel('bottom', 'Energy', units='eV')
# p.setLabel('left', 'Count')

win.show()
QtGui.QApplication.instance().exec_()