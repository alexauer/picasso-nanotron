"""
    gui/nanotron
    ~~~~~~~~~~~~~~~~~~~~
    Graphical user interface for segmentation using deep learning
    :author: Alexander Auer, 2019
    :copyright: Copyright (c) 2016 Jungmann Lab, MPI of Biochemistry
"""
import functools
import multiprocessing
import os.path
import os
import sys
import time
import traceback
from multiprocessing import sharedctypes
from tqdm import tqdm

import matplotlib.pyplot as plt
import numba
import numpy as np
import scipy
import joblib
import yaml
from PyQt5 import QtCore, QtGui, QtWidgets

from .. import io, lib, render, nanotron

DEFAULT_OVERSAMPLING = 1.0
DEFAULT_MODEL_PATH = "/picasso/model/model.sav"

@numba.jit(nopython=True, nogil=True)
def render_hist(x, y, oversampling, t_min, t_max):
    n_pixel = int(np.ceil(oversampling * (t_max - t_min)))
    in_view = (x > t_min) & (y > t_min) & (x < t_max) & (y < t_max)
    x = x[in_view]
    y = y[in_view]
    x = oversampling * (x - t_min)
    y = oversampling * (y - t_min)
    image = np.zeros((n_pixel, n_pixel), dtype=np.float32)
    render._fill(image, x, y)
    return len(x), image

class Worker(QtCore.QThread):

    progressMade = QtCore.pyqtSignal(int, int) # (current pick, total picks, locs)
    finished = QtCore.pyqtSignal(np.recarray)

    def __init__(self, mlp, locs, pick_radius, oversampling, parent=None):
        super().__init__()
        self.model = mlp
        self.locs = locs.copy()
        self.pick_radius = pick_radius
        self.oversampling = oversampling


    def run(self):

        img_shape = int(2 * self.pick_radius * self.oversampling)

        self.prediction = np.zeros(len(np.unique(self.locs['group'])), dtype=[('group','u4'),('prediction','i4'),('score','f4')])
        self.prediction['group'] = np.unique(self.locs['group'])
        len_groups = len(np.unique(self.locs['group']))
        p_locs = np.zeros(len(self.locs['group']), dtype=[('group','u4'),('prediction','i4'),('score','f4')])

        for id, pick in enumerate(tqdm(self.prediction['group'], desc='Predict')):

            self.progressMade.emit(pick, len_groups)

            pred, pred_proba = nanotron.predict_structure(mlp=self.model,
            locs=self.locs, pick=pick, img_shape=img_shape, pick_radius=self.pick_radius,
            oversampling=self.oversampling)

            # Save predictions and scores in numpy array
            self.prediction[self.prediction['group'] == pick] = pick, pred[0], pred_proba.max()
            p_locs[self.locs['group'] == pick] = pick, pred[0], pred_proba.max()

        self.locs = lib.append_to_rec(self.locs, p_locs['prediction'],'prediction')
        self.locs = lib.append_to_rec(self.locs, p_locs['score'],'score')

        self.finished.emit(self.locs)



class ParametersDialog(QtWidgets.QDialog):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Parameters")
        self.setModal(False)
        grid = QtWidgets.QGridLayout(self)

        grid.addWidget(QtWidgets.QLabel("Oversampling:"), 0, 0)
        self.oversampling = QtWidgets.QDoubleSpinBox()
        self.oversampling.setRange(1, 200)
        self.oversampling.setValue(DEFAULT_OVERSAMPLING)
        self.oversampling.setDecimals(1)
        self.oversampling.setKeyboardTracking(False)
        self.oversampling.valueChanged.connect(self.window.update_image)
        grid.addWidget(self.oversampling, 0, 1)

        grid.addWidget(QtWidgets.QLabel("Iterations:"), 1, 0)
        self.iterations = QtWidgets.QSpinBox()
        self.iterations.setRange(0, 1e7)
        self.iterations.setValue(10)
        grid.addWidget(self.iterations, 1, 1)


class View(QtWidgets.QLabel):
    def __init__(self, window):
        super().__init__()
        self.window = window
        self.setMinimumSize(1, 1)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setAcceptDrops(True)
        self._pixmap = None

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        path = urls[0].toLocalFile()
        ext = os.path.splitext(path)[1].lower()
        if ext == ".hdf5":
            self.open(path)

    def resizeEvent(self, event):
        if self._pixmap is not None:
            self.set_pixmap(self._pixmap)

    def set_image(self, image):
        cmap = np.uint8(np.round(255 * plt.get_cmap("hot")(np.arange(256))))
        image /= image.max()
        image = np.minimum(image, 1.0)
        image = np.round(255 * image).astype("uint8")
        Y, X = image.shape
        self._bgra = np.zeros((Y, X, 4), dtype=np.uint8, order="C")
        self._bgra[..., 0] = cmap[:, 2][image]
        self._bgra[..., 1] = cmap[:, 1][image]
        self._bgra[..., 2] = cmap[:, 0][image]
        qimage = QtGui.QImage(self._bgra.data, X, Y, QtGui.QImage.Format_RGB32)
        self._pixmap = QtGui.QPixmap.fromImage(qimage)
        self.set_pixmap(self._pixmap)

    def set_pixmap(self, pixmap):
        self.setPixmap(
            pixmap.scaled(
                self.width(),
                self.height(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.FastTransformation,
            )
        )

    def update_image(self, *args):
        oversampling = self.window.parameters_dialog.oversampling.value()
        t_min = np.min([np.min(self.locs.x), np.min(self.locs.y)])
        t_max = np.max([np.max(self.locs.x), np.max(self.locs.y)])
        N_avg, image_avg = render.render_hist(
            self.locs, oversampling, t_min, t_min, t_max, t_max
        )
        self.set_image(image_avg)


class Window(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Picasso: Nanotron")
        self.resize(768,512)
        this_directory = os.path.dirname(os.path.realpath(__file__))
        icon_path = os.path.join(this_directory, "icons", "nanotron.ico")
        icon = QtGui.QIcon(icon_path)
        self.setWindowIcon(icon)
        self.setAcceptDrops(True)
        self.predicting = False
        self.model_loaded = False

        # self.parameters_dialog = ParametersDialog(self)
        menu_bar = self.menuBar()

        #File menu
        file_menu = menu_bar.addMenu("File")
        open_action = file_menu.addAction("Open")
        open_action.setShortcut(QtGui.QKeySequence.Open)
        open_action.triggered.connect(self.open)
        file_menu.addAction(open_action)

        #Training menu #TODO
        # process_menu = menu_bar.addMenu("Training")
        # parameters_action = process_menu.addAction("Parameters")
        # parameters_action.setShortcut("Ctrl+P")
        # parameters_action.triggered.connect(self.parameters_dialog.show)
        # average_action = process_menu.addAction("Average")
        # average_action.setShortcut("Ctrl+A")
        # average_action.triggered.connect(self.view.average)

        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Load localization file.")
        self.grid = QtWidgets.QGridLayout() # parent grid, all widget were placed in this grid.
        # self.grid.setSpacing(5)

        self.view = QtWidgets.QLabel("")
        minsize = 512
        self.view.setFixedWidth(minsize)
        self.view.setFixedHeight(minsize)

        view_box = QtWidgets.QGroupBox() #render box, locs file is rendered here
        view_grid = QtWidgets.QGridLayout(view_box)
        view_grid.addWidget(self.view,0,0)

        #Model box
        self.load_default_model()
        model_box = QtWidgets.QGroupBox("Model")
        modelbox_grid = QtWidgets.QVBoxLayout(model_box)
        self.model_load_btn = QtWidgets.QPushButton("Load Model")
        modelbox_grid.addWidget(self.model_load_btn)
        self.model_load_btn.clicked.connect(self.load_model)

        #Classes box
        self.class_box = QtWidgets.QGroupBox("Structures") #model box, select what origamis should be exported
        self.classbox_grid = QtWidgets.QVBoxLayout(self.class_box)
        self.update_class_buttons()
        self.classbox_grid.addStretch(1)

        #Predict box
        predict_box = QtWidgets.QGroupBox("Predict")
        predict_grid = QtWidgets.QVBoxLayout(predict_box)
        self.predict_btn = QtWidgets.QPushButton("Predict")
        self.predict_btn.clicked.connect(self.predict)
        predict_grid.addWidget(self.predict_btn)

        accuracy_box = QtWidgets.QGroupBox("Filter export")
        accuracy_grid = QtWidgets.QGridLayout(accuracy_box)
        self.filter_accuracy_btn = QtWidgets.QCheckBox("Accuracy Filter")
        self.export_accuracy = QtWidgets.QDoubleSpinBox()
        self.export_accuracy.setDecimals(2)
        self.export_accuracy.setRange(0, 1)
        self.export_accuracy.setValue(0.9)
        self.export_accuracy.setSingleStep(0.01)
        accuracy_grid.addWidget(self.filter_accuracy_btn,1,0)
        accuracy_grid.addWidget(self.export_accuracy,0,1)

        self.export_btn = QtWidgets.QPushButton("Export")
        self.export_btn.clicked.connect(self.export)

        #Export box
        export_box = QtWidgets.QGroupBox("Export")
        export_grid = QtWidgets.QGridLayout(export_box)
        export_grid.addWidget(self.export_accuracy, 0,1,1,1)
        export_grid.addWidget(self.filter_accuracy_btn,0,0,1,1)
        export_grid.addWidget(self.export_btn,1,0,1,2)

        self.grid.addWidget(view_box,0,0,-3,1)
        self.grid.addWidget(model_box,0,1,1,1)
        self.grid.addWidget(self.class_box,1,1,1,1)
        self.grid.addWidget(predict_box,2,1,1,1)
        self.grid.addWidget(export_box,3,1,1,1)

        mainWidget = QtWidgets.QWidget()
        mainWidget.setLayout(self.grid)
        self.setCentralWidget(mainWidget)


    def predict(self):

        if (self.predicting == False) and (self.model_loaded == True):

            self.predicting = True

            self.oversampling = self.model_info["Oversampling"]
            self.pick_diameter = self.model_info["Pick Diameter"]
            self.pick_radius = self.pick_diameter / 2

            self.thread = Worker(
                self.model, self.locs, self.pick_radius, self.oversampling,
            )
            self.thread.progressMade.connect(self.on_progress)

            self.thread.finished.connect(self.on_finished)
            self.thread.start()

    def on_finished(self, locs):
        self.locs = locs.copy()
        self.predicting = False

    def on_progress(self, pick, total_picks):
        # self.locs = locs.copy()
        self.status_bar.showMessage(
            "From {} picks - predicted {}".format(total_picks,pick)
        )

    def open(self):
        path, exe = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open localizations", filter="*.hdf5"
        )
        if path:
            self.openFile(path)

    def openFile(self, path):
        self.path = path

        try:
            self.locs, self.info = io.load_locs(path, qt_parent=self)
        except io.NoMetadataFileError:
            return

        if not hasattr(self.locs, "group"):
            msgBox = QtWidgets.QMessageBox(self)
            msgBox.setWindowTitle("Error")
            msgBox.setText(
                ("Datafile does not contain group information."
                    " Please load file with picked localizations.")
            )
            msgBox.exec_()
        else:
            groups = np.unique(self.locs.group)
            groups_max = max(groups)
            n_locs = len(self.locs)

            self.update_image()
            self.status_bar.showMessage("{} picks loaded. Ready for processing.".format(str(groups_max)))

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        path = urls[0].toLocalFile()
        ext = os.path.splitext(path)[1].lower()
        if ext == ".hdf5":
            print("Opening {} ..".format(path))
            self.openFile(path)

    def update_image(self, *args):

        oversampling = 1
        t_min = np.min([np.min(self.locs.x), np.min(self.locs.y)])
        t_max = np.max([np.max(self.locs.x), np.max(self.locs.y)])
        N_avg, image = render.render_hist(
            self.locs, oversampling, t_min, t_min, t_max, t_max
        )
        self.set_image(image)

    def set_pixmap(self, pixmap):
        self.view.setPixmap(
            pixmap.scaled(
                self.width(),
                self.height(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.FastTransformation,
            )
        )

    def set_image(self, image):
        cmap = np.uint8(np.round(255 * plt.get_cmap("hot")(np.arange(256))))
        image /= image.max()
        image = np.minimum(image, 1.0)
        image = np.round(255 * image).astype("uint8")
        Y, X = image.shape
        self._bgra = np.zeros((Y, X, 4), dtype=np.uint8, order="C")
        self._bgra[..., 0] = cmap[:, 2][image]
        self._bgra[..., 1] = cmap[:, 1][image]
        self._bgra[..., 2] = cmap[:, 0][image]
        qimage = QtGui.QImage(self._bgra.data, X, Y, QtGui.QImage.Format_RGB32)
        qimage = qimage.scaled(
            self.view.width(),
            np.round(self.view.height() * Y / X),
            QtCore.Qt.KeepAspectRatioByExpanding,
        )
        self._pixmap = QtGui.QPixmap.fromImage(qimage)
        self.set_pixmap(self._pixmap)

    def load_default_model(self):

        path = os.getcwd() + DEFAULT_MODEL_PATH

        try:
            self.model = joblib.load(path)
        except Exception as e:
            raise ValueError("No model file loaded.")

        try:
            with open(path[:-3]+'yaml', "r") as f:
                self.model_info = yaml.load(f, Loader = yaml.FullLoader)
                self.classes = []
                self.classes = self.model_info["Classes"]
                self.model_loaded = True
        except Exception as e:
            raise ValueError("No classes in model metadata file.")


    def load_model(self):

        path, exe = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load model file", directory=None)
        if path:

            try:
                self.model = joblib.load(path)
            except Exception as e:
                raise ValueError("No model file loaded.")

            try:
                with open(path[:-3]+'yaml', "r") as f:
                    self.model_info = yaml.load(f, Loader = yaml.FullLoader)
                    self.classes = []
                    self.classes = self.model_info["Classes"]
                    self.model_loaded = True
            except Exception as e:
                raise ValueError("No classes in model metadata file.")

    def update_class_buttons(self):

        for id, name in self.classes.items():
            c = QtWidgets.QCheckBox(name)
            self.classbox_grid.addWidget(c)

    def export(self):

        if not hasattr(self.locs, "prediction"):
            msgBox = QtWidgets.QMessageBox(self)
            msgBox.setWindowTitle("Error")
            msgBox.setText("No predictions. Predict first.")
            msgBox.exec_()
        else:
            export_map = []
            export_classes = {}

            checks = (self.classbox_grid.itemAt(i) for i in range(self.classbox_grid.count()))
            for btn in checks:

                if isinstance(btn, QtWidgets.QWidgetItem):
                    if btn.widget().checkState():
                        export_map.append(True)
                    else:
                        export_map.append(False)

            for key, item in self.classes.items():
                if export_map[key] == True:
                    export_classes[key] = item


            accuracy = self.export_accuracy.value()

            if self.filter_accuracy_btn.isChecked():
                filtering = True
            else:
                filtering = False

            nanotron.export_locs(locs = self.locs, path = self.path, classes=self.classes, filtering=filtering,
            accuracy = accuracy, regroup = True)


def main():

    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.show()

    def excepthook(type, value, tback):
        lib.cancel_dialogs()
        message = "".join(traceback.format_exception(type, value, tback))
        errorbox = QtWidgets.QMessageBox.critical(
            window, "An error occured", message
        )
        errorbox.exec_()
        sys.__excepthook__(type, value, tback)

    sys.excepthook = excepthook

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
