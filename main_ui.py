# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Users\dei\Documents\Programming\ZZZ Traffic sign THESIS\main.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(770, 561)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.Vid_label = QtWidgets.QLabel(self.centralwidget)
        self.Vid_label.setText("")
        self.Vid_label.setObjectName("Vid_label")
        self.verticalLayout.addWidget(self.Vid_label)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.scrollArea.sizePolicy().hasHeightForWidth())
        self.scrollArea.setSizePolicy(sizePolicy)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 369, 466))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.verticalLayout_2.addWidget(self.scrollArea)
        self.Inference_Button = QtWidgets.QPushButton(self.centralwidget)
        self.Inference_Button.setObjectName("Inference_Button")
        self.verticalLayout_2.addWidget(self.Inference_Button)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 770, 22))
        self.menubar.setObjectName("menubar")
        self.menuCamera = QtWidgets.QMenu(self.menubar)
        self.menuCamera.setObjectName("menuCamera")
        self.menuView = QtWidgets.QMenu(self.menubar)
        self.menuView.setObjectName("menuView")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        self.menuModels = QtWidgets.QMenu(self.menubar)
        self.menuModels.setObjectName("menuModels")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setSizeGripEnabled(False)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionRefresh = QtWidgets.QAction(MainWindow)
        self.actionRefresh.setObjectName("actionRefresh")
        self.actionHi = QtWidgets.QAction(MainWindow)
        self.actionHi.setObjectName("actionHi")
        self.actionVideo_Inference = QtWidgets.QAction(MainWindow)
        self.actionVideo_Inference.setObjectName("actionVideo_Inference")
        self.actionSet_YOLO_Model = QtWidgets.QAction(MainWindow)
        self.actionSet_YOLO_Model.setObjectName("actionSet_YOLO_Model")
        self.actionSet_ResNet50V2_Model = QtWidgets.QAction(MainWindow)
        self.actionSet_ResNet50V2_Model.setObjectName("actionSet_ResNet50V2_Model")
        self.actionLoad_YOLO = QtWidgets.QAction(MainWindow)
        self.actionLoad_YOLO.setObjectName("actionLoad_YOLO")
        self.actionLoad_ResNet = QtWidgets.QAction(MainWindow)
        self.actionLoad_ResNet.setObjectName("actionLoad_ResNet")
        self.menuCamera.addSeparator()
        self.menuCamera.addAction(self.actionRefresh)
        self.menuModels.addAction(self.actionLoad_YOLO)
        self.menuModels.addAction(self.actionLoad_ResNet)
        self.menubar.addAction(self.menuCamera.menuAction())
        self.menubar.addAction(self.menuModels.menuAction())
        self.menubar.addAction(self.menuView.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Inference_Button.setText(_translate("MainWindow", "Enable YOLO_ResNet50V2 Inference"))
        self.menuCamera.setTitle(_translate("MainWindow", "Camera"))
        self.menuView.setTitle(_translate("MainWindow", "View"))
        self.menuHelp.setTitle(_translate("MainWindow", "Help"))
        self.menuModels.setTitle(_translate("MainWindow", "Models"))
        self.actionRefresh.setText(_translate("MainWindow", "Refresh"))
        self.actionHi.setText(_translate("MainWindow", "Hi"))
        self.actionVideo_Inference.setText(_translate("MainWindow", "Video Inference"))
        self.actionSet_YOLO_Model.setText(_translate("MainWindow", "Set YOLO Model"))
        self.actionSet_ResNet50V2_Model.setText(_translate("MainWindow", "Set ResNet50V2 Model"))
        self.actionLoad_YOLO.setText(_translate("MainWindow", "Load a YOLO Model"))
        self.actionLoad_ResNet.setText(_translate("MainWindow", "Load a ResNet50v2 Model"))
