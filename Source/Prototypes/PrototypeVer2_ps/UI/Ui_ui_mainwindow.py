# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'ui_mainwindow.ui'
##
## Created by: Qt User Interface Compiler version 5.15.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import (QCoreApplication, QDate, QDateTime, QMetaObject,
    QObject, QPoint, QRect, QSize, QTime, QUrl, Qt)
from PySide2.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont,
    QFontDatabase, QIcon, QKeySequence, QLinearGradient, QPalette, QPainter,
    QPixmap, QRadialGradient)
from PySide2.QtWidgets import *
from UI.item_widget import ItemPlotWidget

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.setWindowModality(Qt.NonModal)
        MainWindow.resize(1061, 605)
        MainWindow.setAnimated(False)
        MainWindow.setDocumentMode(False)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.centralwidget.setLayoutDirection(Qt.LeftToRight)
        self.horizontalLayout_4 = QHBoxLayout(self.centralwidget)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.wdt_history = ItemPlotWidget(self.centralwidget)
        self.wdt_history.setObjectName(u"wdt_history")

        self.horizontalLayout_2.addWidget(self.wdt_history)

        self.horizontalLayout_2.setStretch(0, 10)

        self.verticalLayout_2.addLayout(self.horizontalLayout_2)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.groupBox_2 = QGroupBox(self.centralwidget)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.horizontalLayout_5 = QHBoxLayout(self.groupBox_2)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.wdt_interest_rates = ItemPlotWidget(self.groupBox_2)
        self.wdt_interest_rates.setObjectName(u"wdt_interest_rates")

        self.horizontalLayout_5.addWidget(self.wdt_interest_rates)


        self.horizontalLayout_3.addWidget(self.groupBox_2)

        self.groupBox_3 = QGroupBox(self.centralwidget)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.horizontalLayout_6 = QHBoxLayout(self.groupBox_3)
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.wdt_gdp = ItemPlotWidget(self.groupBox_3)
        self.wdt_gdp.setObjectName(u"wdt_gdp")

        self.horizontalLayout_6.addWidget(self.wdt_gdp)


        self.horizontalLayout_3.addWidget(self.groupBox_3)

        self.groupBox_4 = QGroupBox(self.centralwidget)
        self.groupBox_4.setObjectName(u"groupBox_4")
        self.horizontalLayout_7 = QHBoxLayout(self.groupBox_4)
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.wdt_dem_after_shock = ItemPlotWidget(self.groupBox_4)
        self.wdt_dem_after_shock.setObjectName(u"wdt_dem_after_shock")

        self.horizontalLayout_7.addWidget(self.wdt_dem_after_shock)


        self.horizontalLayout_3.addWidget(self.groupBox_4)

        self.groupBox_7 = QGroupBox(self.centralwidget)
        self.groupBox_7.setObjectName(u"groupBox_7")
        self.horizontalLayout_9 = QHBoxLayout(self.groupBox_7)
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.wdt_price_lvl = ItemPlotWidget(self.groupBox_7)
        self.wdt_price_lvl.setObjectName(u"wdt_price_lvl")

        self.horizontalLayout_9.addWidget(self.wdt_price_lvl)


        self.horizontalLayout_3.addWidget(self.groupBox_7)

        self.groupBox_6 = QGroupBox(self.centralwidget)
        self.groupBox_6.setObjectName(u"groupBox_6")
        self.horizontalLayout_8 = QHBoxLayout(self.groupBox_6)
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.wdt_delta_price_lvl = ItemPlotWidget(self.groupBox_6)
        self.wdt_delta_price_lvl.setObjectName(u"wdt_delta_price_lvl")

        self.horizontalLayout_8.addWidget(self.wdt_delta_price_lvl)


        self.horizontalLayout_3.addWidget(self.groupBox_6)


        self.verticalLayout_2.addLayout(self.horizontalLayout_3)

        self.verticalLayout_2.setStretch(0, 6)
        self.verticalLayout_2.setStretch(1, 4)

        self.horizontalLayout_4.addLayout(self.verticalLayout_2)

        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")
        self.label.setMaximumSize(QSize(40, 16777215))

        self.horizontalLayout.addWidget(self.label)

        self.cmbFilter = QComboBox(self.centralwidget)
        self.cmbFilter.setObjectName(u"cmbFilter")

        self.horizontalLayout.addWidget(self.cmbFilter)

        self.btnPlay = QPushButton(self.centralwidget)
        self.btnPlay.setObjectName(u"btnPlay")
        self.btnPlay.setMinimumSize(QSize(0, 30))
        self.btnPlay.setMaximumSize(QSize(50, 16777215))

        self.horizontalLayout.addWidget(self.btnPlay)


        self.verticalLayout_3.addLayout(self.horizontalLayout)

        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.groupBox_9 = QGroupBox(self.centralwidget)
        self.groupBox_9.setObjectName(u"groupBox_9")
        self.horizontalLayout_11 = QHBoxLayout(self.groupBox_9)
        self.horizontalLayout_11.setObjectName(u"horizontalLayout_11")
        self.wdt_affinity = ItemPlotWidget(self.groupBox_9)
        self.wdt_affinity.setObjectName(u"wdt_affinity")

        self.horizontalLayout_11.addWidget(self.wdt_affinity)


        self.verticalLayout.addWidget(self.groupBox_9)

        self.groupBox_8 = QGroupBox(self.centralwidget)
        self.groupBox_8.setObjectName(u"groupBox_8")
        self.horizontalLayout_10 = QHBoxLayout(self.groupBox_8)
        self.horizontalLayout_10.setObjectName(u"horizontalLayout_10")
        self.wdt_delta_affinity = ItemPlotWidget(self.groupBox_8)
        self.wdt_delta_affinity.setObjectName(u"wdt_delta_affinity")

        self.horizontalLayout_10.addWidget(self.wdt_delta_affinity)


        self.verticalLayout.addWidget(self.groupBox_8)


        self.verticalLayout_3.addLayout(self.verticalLayout)

        self.verticalLayout_3.setStretch(0, 1)
        self.verticalLayout_3.setStretch(1, 90)

        self.horizontalLayout_4.addLayout(self.verticalLayout_3)

        self.horizontalLayout_4.setStretch(0, 7)
        self.horizontalLayout_4.setStretch(1, 3)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Simulation", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("MainWindow", u"Interest Rate:", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("MainWindow", u"GDP:", None))
        self.groupBox_4.setTitle(QCoreApplication.translate("MainWindow", u"Dem_After_Shock:", None))
        self.groupBox_7.setTitle(QCoreApplication.translate("MainWindow", u"Price_LVL:", None))
        self.groupBox_6.setTitle(QCoreApplication.translate("MainWindow", u"Delta_Price_LVL:", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Show:", None))
        self.btnPlay.setText(QCoreApplication.translate("MainWindow", u"Pause", None))
        self.groupBox_9.setTitle(QCoreApplication.translate("MainWindow", u"Affinity:", None))
        self.groupBox_8.setTitle(QCoreApplication.translate("MainWindow", u"Delta Affinity:", None))
    # retranslateUi

