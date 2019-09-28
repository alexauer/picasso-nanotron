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
import sys
import time
import traceback
from multiprocessing import sharedctypes

import matplotlib.pyplot as plt
import numba
import numpy as np
import scipy
from PyQt5 import QtCore, QtGui, QtWidgets

from .. import io, lib, render

DEFAULT_OVERSAMPLING = 1.0

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


def compute_xcorr(CF_image_avg, image):
    F_image = np.fft.fft2(image)
    xcorr = np.fft.fftshift(np.real(np.fft.ifft2((F_image * CF_image_avg))))
    return xcorr


def align_group(
    angles,
    oversampling,
    t_min,
    t_max,
    CF_image_avg,
    image_half,
    counter,
    lock,
    group,
):
    with lock:
        counter.value += 1
    index = group_index[group].nonzero()[1]
    x_rot = x[index]
    y_rot = y[index]
    x_original = x_rot.copy()
    y_original = y_rot.copy()
    xcorr_max = 0.0
    for angle in angles:
        # rotate locs
        x_rot = np.cos(angle) * x_original - np.sin(angle) * y_original
        y_rot = np.sin(angle) * x_original + np.cos(angle) * y_original
        # render group image
        N, image = render_hist(x_rot, y_rot, oversampling, t_min, t_max)
        # calculate cross-correlation
        xcorr = compute_xcorr(CF_image_avg, image)
        # find the brightest pixel
        y_max, x_max = np.unravel_index(xcorr.argmax(), xcorr.shape)
        # store the transformation if the correlation is larger than before
        if xcorr[y_max, x_max] > xcorr_max:
            xcorr_max = xcorr[y_max, x_max]
            rot = angle
            dy = np.ceil(y_max - image_half) / oversampling
            dx = np.ceil(x_max - image_half) / oversampling
    # rotate and shift image group locs
    x[index] = np.cos(rot) * x_original - np.sin(rot) * y_original - dx
    y[index] = np.sin(rot) * x_original + np.cos(rot) * y_original - dy


def init_pool(x_, y_, group_index_):
    global x, y, group_index
    x = np.ctypeslib.as_array(x_)
    y = np.ctypeslib.as_array(y_)
    group_index = group_index_


class Worker(QtCore.QThread):

    progressMade = QtCore.pyqtSignal(int, int, int, int, np.recarray, bool)

    def __init__(self, locs, r, group_index, oversampling, iterations):
        super().__init__()
        self.locs = locs.copy()
        self.r = r
        self.t_min = -r
        self.t_max = r
        self.group_index = group_index
        self.oversampling = oversampling
        self.iterations = iterations

    def run(self):
        n_groups = self.group_index.shape[0]
        a_step = np.arcsin(1 / (self.oversampling * self.r))
        angles = np.arange(0, 2 * np.pi, a_step)
        n_workers = max(1, int(0.75 * multiprocessing.cpu_count()))
        manager = multiprocessing.Manager()
        counter = manager.Value("d", 0)
        lock = manager.Lock()
        groups_per_worker = max(1, int(n_groups / n_workers))
        for it in range(self.iterations):
            counter.value = 0
            # render average image
            N_avg, image_avg = render.render_hist(
                self.locs,
                self.oversampling,
                self.t_min,
                self.t_min,
                self.t_max,
                self.t_max,
            )
            n_pixel, _ = image_avg.shape
            image_half = n_pixel / 2
            CF_image_avg = np.conj(np.fft.fft2(image_avg))
            # TODO: blur average
            fc = functools.partial(
                align_group,
                angles,
                self.oversampling,
                self.t_min,
                self.t_max,
                CF_image_avg,
                image_half,
                counter,
                lock,
            )
            result = pool.map_async(fc, range(n_groups), groups_per_worker)
            while not result.ready():
                self.progressMade.emit(
                    it + 1,
                    self.iterations,
                    counter.value,
                    n_groups,
                    self.locs,
                    False,
                )
                time.sleep(0.5)
            self.locs.x = np.ctypeslib.as_array(x)
            self.locs.y = np.ctypeslib.as_array(y)
            self.locs.x -= np.mean(self.locs.x)
            self.locs.y -= np.mean(self.locs.y)
            self.progressMade.emit(
                it + 1,
                self.iterations,
                counter.value,
                n_groups,
                self.locs,
                True,
            )


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
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Picasso: Nanotron")
        self.resize(768,512)
        this_directory = os.path.dirname(os.path.realpath(__file__))
        icon_path = os.path.join(this_directory, "icons", "nanotron.ico")
        icon = QtGui.QIcon(icon_path)
        self.setWindowIcon(icon)
        self.setAcceptDrops(True)

        # self.parameters_dialog = ParametersDialog(self)
        menu_bar = self.menuBar()

        #File menu
        file_menu = menu_bar.addMenu("File")
        open_action = file_menu.addAction("Open")
        open_action.setShortcut(QtGui.QKeySequence.Open)
        open_action.triggered.connect(self.open)
        file_menu.addAction(open_action)
        save_action = file_menu.addAction("Save")
        save_action.setShortcut(QtGui.QKeySequence.Save)
        save_action.triggered.connect(self.save)
        file_menu.addAction(save_action)

        #Training menu #TODO
        # process_menu = menu_bar.addMenu("Training")
        # parameters_action = process_menu.addAction("Parameters")
        # parameters_action.setShortcut("Ctrl+P")
        # parameters_action.triggered.connect(self.parameters_dialog.show)
        # average_action = process_menu.addAction("Average")
        # average_action.setShortcut("Ctrl+A")
        # average_action.triggered.connect(self.view.average)

        self.status_bar = self.statusBar()

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
        model_box = QtWidgets.QGroupBox("Select Classes") #model box, select what origamis should be exported
        modelbox_grid = QtWidgets.QVBoxLayout(model_box)
        self.model_0_btn = QtWidgets.QCheckBox("Digit 1")
        self.model_1_btn = QtWidgets.QCheckBox("Digit 2")
        self.model_2_btn = QtWidgets.QCheckBox("20 nm Grid")
        # self.model_3_btn = QtWidgets.QCheckBox("4 Corner") #Todo
        modelbox_grid.addWidget(self.model_0_btn)
        modelbox_grid.addWidget(self.model_1_btn)
        modelbox_grid.addWidget(self.model_2_btn)
        modelbox_grid.addWidget(self.model_3_btn)
        modelbox_grid.addStretch(1)

        #Predict box
        predict_box = QtWidgets.QGroupBox("Predict")
        predict_grid = QtWidgets.QVBoxLayout(predict_box)
        self.predict_btn = QtWidgets.QPushButton("Predict")
        predict_grid.addWidget(self.predict_btn)

        accuracy_box = QtWidgets.QGroupBox("Filter export")
        accuracy_grid = QtWidgets.QGridLayout(accuracy_box)
        self.filter_accuracy_btn = QtWidgets.QCheckBox("Filter")
        self.export_accuracy = QtWidgets.QDoubleSpinBox()
        self.export_accuracy.setDecimals(2)
        self.export_accuracy.setRange(0, 1)
        self.export_accuracy.setValue(0.9)
        self.export_accuracy.setSingleStep(0.01)
        accuracy_grid.addWidget(self.filter_accuracy_btn,1,0)
        accuracy_grid.addWidget(self.export_accuracy,0,1)

        self.export_btn = QtWidgets.QPushButton("Export")

        #Export box
        export_box = QtWidgets.QGroupBox("Export")
        export_grid = QtWidgets.QGridLayout(export_box)
        export_grid.addWidget(self.export_accuracy, 0,1,1,1)
        export_grid.addWidget(self.filter_accuracy_btn,0,0,1,1)
        export_grid.addWidget(self.export_btn,1,0,1,2)

        self.grid.addWidget(view_box,0,0,-2,1)
        self.grid.addWidget(model_box,0,1,1,1)
        self.grid.addWidget(predict_box,1,1,1,1)
        self.grid.addWidget(export_box,2,1,1,1)

        mainWidget = QtWidgets.QWidget()
        mainWidget.setLayout(self.grid)
        self.setCentralWidget(mainWidget)

    def open(self):
        path, exe = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open localizations", filter="*.hdf5"
        )
        if path:
            self.openFile(path)

    def save(self):
        out_path = os.path.splitext(self.view.path)[0] + "_avg.hdf5"
        path, exe = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save localizations", out_path, filter="*.hdf5"
        )
        if path:
            self.view.save(path)

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
