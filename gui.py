import numpy as np
from PyQt5 import uic, QtGui
import pyqtgraph.opengl as gl
import pyqtgraph as pg
from inspect import getargspec, signature
from materials import materials_list
from subjects import Space, subjects_list

pg.mkQApp() 


mainWindow = uic.loadUi("UI/gui.ui")
spaceView = mainWindow.graphicsView


def get_space_size():
    global mainWindow
    size = np.array([
        mainWindow.doubleSpinBoxOfXSize.value(),
        mainWindow.doubleSpinBoxOfYSize.value(),
        mainWindow.doubleSpinBoxOfZSize.value()
    ])
    return size


def update_space():
    global spaceView, space_axis, get_space_size
    space_size = get_space_size()
    space_box.setSize(*space_size)
    spaceView.setCameraPosition(
        pos=QtGui.QVector3D(*(space_size/2))
        )
    space_axis.setSize(*(space_size + space_size*0.1))


def update_comboBoxOfMaterials():
    mainWindow.comboBoxOfMaterials.clear()
    mainWindow.comboBoxOfMaterials.addItems(materials_list)


def adding_subject():
    addingSubjectWindow = uic.loadUi("UI/addingSubject.ui")
    addingSubjectWindow.comboBox_subjects.addItems(subjects_list)
    addingSubjectWindow.exec()
    if addingSubjectWindow.result():
        # subject = subjects_list[addingSubjectWindow.comboBox_subjects.currentText()]
        # print(subject)
        name = addingSubjectWindow.lineEdit_name.text()
        mainWindow.listWidget_subjects.addItem(name)
        currentWidget = addingSubjectWindow.stackedWidget_subjectProperties.currentWidget()
        mainWindow.stackedWidget_subjectProperties.addWidget(currentWidget)


space_box = gl.GLBoxItem()
spaceView.addItem(space_box)
spaceView.setCameraPosition(
    pos=QtGui.QVector3D(*(get_space_size()/2)),
    distance=np.max(get_space_size())*3
    )
space_axis = gl.GLAxisItem()
space_axis.setDepthValue(-2)
spaceView.addItem(space_axis)


# phantom = np.load('/home/mikhail/Documents/Projects Python/SPECT Modeling/efg3cut_phantom.npy')
# phantom = Phantom(
#     coordinates=(0., 7., 0.),
#     material=phantom,
#     voxel_size=0.4)


# mainWindow.listWidget_subjects.addItems(subjects)

mainWindow.doubleSpinBoxOfXSize.valueChanged.connect(update_space)
mainWindow.doubleSpinBoxOfYSize.valueChanged.connect(update_space)
mainWindow.doubleSpinBoxOfZSize.valueChanged.connect(update_space)

mainWindow.pushButton_addSubject.clicked.connect(adding_subject)

update_comboBoxOfMaterials()
update_space()
mainWindow.show()
QtGui.QApplication.exec()