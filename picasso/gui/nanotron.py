"""
    gui/nanotron
    ~~~~~~~~~~~~~~~~~~~~
    Graphical user interface for segmentation using deep learning
    :author: Alexander Auer, 2019
    :copyright: Copyright (c) 2016 Jungmann Lab, MPI of Biochemistry
"""
import os.path
import os
import sys
import traceback
from tqdm import tqdm

import matplotlib.pyplot as plt
import numba
import numpy as np
import joblib
import yaml
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon

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


class Trainer(QtCore.QThread):

    trainings_made = QtCore.pyqtSignal(int, int)
    training_finished = QtCore.pyqtSignal(np.recarray)

    def __init__(self, mlp, locs, pick_radius, oversampling, parent=None):
        super().__init__()
        self.model = mlp
        self.locs = locs.copy()
        self.pick_radius = pick_radius
        self.oversampling = oversampling

    def train(self):

        return

    def validate(self):

        return


class Predicter(QtCore.QThread):

    predictions_made = QtCore.pyqtSignal(int, int)
    prediction_finished = QtCore.pyqtSignal(np.recarray)

    def __init__(self, mlp, locs, pick_radius, oversampling, parent=None):
        super().__init__()
        self.model = mlp
        self.locs = locs.copy()
        self.pick_radius = pick_radius
        self.oversampling = oversampling

    def run(self):

        img_shape = int(2 * self.pick_radius * self.oversampling)
        self.prediction = np.zeros(len(np.unique(self.locs['group'])),
                                   dtype=[('group', 'u4'),
                                   ('prediction', 'i4'), ('score', 'f4')])
        self.prediction['group'] = np.unique(self.locs['group'])
        len_groups = len(np.unique(self.locs['group']))
        p_locs = np.zeros(len(self.locs['group']), dtype=[('group', 'u4'),
                          ('prediction', 'i4'), ('score', 'f4')])

        for id, pick in enumerate(tqdm(self.prediction['group'],
                                  desc='Predict')):

            self.predictions_made.emit(pick, len_groups)

            pred, pred_proba = nanotron.predict_structure(mlp=self.model,
                                                          locs=self.locs,
                                                          pick=pick,
                                                          img_shape=img_shape,
                                                          pick_radius=self.pick_radius,
                                                          oversampling=self.oversampling)

            # Save predictions and scores in numpy array
            self.prediction[self.prediction['group'] == pick] = pick, pred[0], pred_proba.max()
            p_locs[self.locs['group'] == pick] = pick, pred[0], pred_proba.max()

        self.locs = lib.append_to_rec(self.locs, p_locs['prediction'], 'prediction')
        self.locs = lib.append_to_rec(self.locs, p_locs['score'], 'score')

        self.prediction_finished.emit(self.locs)


class train_dialog(QtWidgets.QDialog):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Train Model")
        self.setModal(False)
        grid = QtWidgets.QGridLayout(self)
        self.file_slots_generated = False
        self.data_prepared = False

        # predict_box = QtWidgets.QGroupBox("Predict")
        # predict_grid = QtWidgets.QVBoxLayout(predict_box)
        # self.predict_btn = QtWidgets.QPushButton("Predict")
        # self.predict_btn.clicked.connect(self.predict)
        # predict_grid.addWidget(self.predict_btn)

        progress_bar = QtWidgets.QProgressBar(self)

        choose_class_box = QtWidgets.QGroupBox("Number of Classes")
        choose_class_grid = QtWidgets.QGridLayout(choose_class_box)
        self.choose_files_n = QtWidgets.QSpinBox()
        self.choose_files_n.setRange(1, 6)
        self.choose_files_n.setValue(0)
        self.choose_files_n.setKeyboardTracking(False)

        file_slots_btn = QtWidgets.QPushButton("Generate Files")
        file_slots_btn.clicked.connect(self.update_train_files)
        choose_class_grid.addWidget(QtWidgets.QLabel("Classes:"), 0, 0)
        choose_class_grid.addWidget(self.choose_files_n, 0, 1)
        choose_class_grid.addWidget(file_slots_btn, 0, 2)

        train_files_box = QtWidgets.QGroupBox("Training Files")
        self.train_files_grid = QtWidgets.QGridLayout(train_files_box)

        prepare_data_btn = QtWidgets.QPushButton("Prepare Data")
        prepare_data_btn.clicked.connect(self.prepare_data)

        perceptron_box = QtWidgets.QGroupBox("Perceptron")
        perceptron_grid = QtWidgets.QGridLayout(perceptron_box)
        perceptron_grid.addWidget(QtWidgets.QLabel("Nodes:"), 0, 0)

        nodes = QtWidgets.QSpinBox()
        nodes.setRange(1, 1000)
        nodes.setValue(100)
        nodes.setKeyboardTracking(False)
        perceptron_grid.addWidget(nodes, 0, 1)

        perceptron_grid.addWidget(QtWidgets.QLabel("Solver:"), 2, 0)
        activation_ft = QtWidgets.QComboBox()
        activation_ft.addItems(['adam', 'lbfgs', 'sgd'])
        perceptron_grid.addWidget(activation_ft, 2, 1)

        perceptron_grid.addWidget(QtWidgets.QLabel("Activation:"), 3, 0)
        activation_ft = QtWidgets.QComboBox()
        activation_ft.addItems(['relu', 'identity', 'logistic', 'tanh'])
        perceptron_grid.addWidget(activation_ft, 3, 1)

        train_parameter_box = QtWidgets.QGroupBox("Training")
        train_parameter_grid = QtWidgets.QGridLayout(train_parameter_box)

        train_parameter_grid.addWidget(QtWidgets.QLabel("Iterations:"), 0, 0)
        self.iterations = QtWidgets.QSpinBox()
        self.iterations.setRange(0, 1e4)
        self.iterations.setValue(200)
        train_parameter_grid.addWidget(self.iterations, 0, 1)

        train_parameter_grid.addWidget(QtWidgets.QLabel("Learning Rate:"), 1, 0)
        self.learing_rate = QtWidgets.QDoubleSpinBox()
        self.learing_rate.setRange(0, 10)
        self.learing_rate.setValue(0.01)
        self.learing_rate.setSingleStep(0.01)
        self.learing_rate.setDecimals(4)
        train_parameter_grid.addWidget(self.learing_rate, 1, 1)

        train_btn = QtWidgets.QPushButton("Train")
        train_btn.clicked.connect(self.train)
        validate_btn = QtWidgets.QPushButton("Validate")
        validate_btn.clicked.connect(self.validate)

        progress_label = QtWidgets.QLabel("")
        progress_label.setAlignment(QtCore.Qt.AlignCenter)

        grid.addWidget(choose_class_box, 0, 0, 1, 1)
        grid.addWidget(train_files_box, 1, 0, 7, 1)
        grid.addWidget(prepare_data_btn, 8, 0, 1, 1)

        grid.addWidget(perceptron_box, 0, 1, 3, 1)
        grid.addWidget(train_parameter_box, 4, 1,)
        grid.addWidget(train_btn, 5, 1, 1, 2)
        grid.addWidget(validate_btn, 6, 1, 1, 2)
        grid.addWidget(progress_label, 7, 1, 1, 2)
        grid.addWidget(progress_bar, 8, 1, 1, 2)

    def train(self):
        # Do
        return

    def validate(self):
        # Do
        return

    def update_train_files(self):

        if not self.file_slots_generated:
            self.file_slots_generated = True

            for file in range(self.choose_files_n.value()):

                c = QtWidgets.QLabel('{}'.format(file))
                self.train_files_grid.addWidget(c, file, 0)

                f = QtWidgets.QPushButton("Load File")
                f.clicked.connect(self.load_train_file())
                self.train_files_grid.addWidget(f, file, 1)

                la = QtWidgets.QLabel("Name:".format(file))
                self.train_files_grid.addWidget(la, file, 2)

                id = QtWidgets.QLineEdit(self)
                id.move(20, 20)
                id.resize(500, 40)
                id.setMaxLength(10)
                self.train_files_grid.addWidget(id, file, 3)

        return

    def load_train_file(self):
        # Do
        print('yay')
        return

    def prepare_data(self):

        return


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
        self.resize(768, 512)
        this_directory = os.path.dirname(os.path.realpath(__file__))
        self.icon_path = os.path.join(this_directory, "icons", "nanotron.ico")
        icon = QtGui.QIcon(self.icon_path)
        self.setWindowIcon(icon)
        self.setAcceptDrops(True)
        self.predicting = False
        self.model_loaded = False
        self.nanotron_log = {}

        # self.parameters_dialog = ParametersDialog(self)
        menu_bar = self.menuBar()
        self.train_dialog = train_dialog(self)

        file_menu = menu_bar.addMenu("File")
        open_action = file_menu.addAction("Open")
        open_action.setShortcut(QtGui.QKeySequence.Open)
        open_action.triggered.connect(self.open)
        file_menu.addAction(open_action)
        export_action = file_menu.addAction("Save")
        export_action.setShortcut(QtGui.QKeySequence.Save)
        export_action.triggered.connect(self.export)
        file_menu.addAction(export_action)

        # Training menu #TODO
        tools_menu = menu_bar.addMenu("Tools")
        load_model_action = tools_menu.addAction("Load Model")
        load_model_action.setShortcut("Ctrl+L")
        load_model_action.triggered.connect(self.load_model)
        train_model_action = tools_menu.addAction("Train Model")
        train_model_action.triggered.connect(self.train_dialog.show)

        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Load localization file.")
        self.grid = QtWidgets.QGridLayout()
        # self.grid.setSpacing(5)

        self.view = QtWidgets.QLabel("")
        minsize = 512
        self.view.setFixedWidth(minsize)
        self.view.setFixedHeight(minsize)

        view_box = QtWidgets.QGroupBox()
        view_grid = QtWidgets.QGridLayout(view_box)
        view_grid.addWidget(self.view, 0, 0)

        self.load_default_model()
        # model_box = QtWidgets.QGroupBox("Model")
        # modelbox_grid = QtWidgets.QVBoxLayout(model_box)
        # self.model_load_btn = QtWidgets.QPushButton("Load Model")
        # modelbox_grid.addWidget(self.model_load_btn)
        # self.model_load_btn.clicked.connect(self.load_model)

        self.class_box = QtWidgets.QGroupBox("Export Structures")
        self.classbox_grid = QtWidgets.QVBoxLayout(self.class_box)
        self.update_class_buttons()
        self.classbox_grid.addStretch(1)

        predict_box = QtWidgets.QGroupBox("Predict")
        predict_grid = QtWidgets.QVBoxLayout(predict_box)
        self.predict_btn = QtWidgets.QPushButton("Predict")
        self.predict_btn.clicked.connect(self.predict)
        predict_grid.addWidget(self.predict_btn)

        accuracy_box = QtWidgets.QGroupBox("Filter export")
        accuracy_grid = QtWidgets.QGridLayout(accuracy_box)
        self.filter_accuracy_btn = QtWidgets.QCheckBox("Probability")
        self.export_accuracy = QtWidgets.QDoubleSpinBox()
        self.export_accuracy.setDecimals(2)
        self.export_accuracy.setRange(0, 1)
        self.export_accuracy.setValue(0.99)
        self.export_accuracy.setSingleStep(0.01)
        accuracy_grid.addWidget(self.filter_accuracy_btn, 1, 0)
        accuracy_grid.addWidget(self.export_accuracy, 0, 1)

        self.export_btn = QtWidgets.QPushButton("Export")
        self.export_btn.clicked.connect(self.export)

        export_box = QtWidgets.QGroupBox("Export")
        export_grid = QtWidgets.QGridLayout(export_box)
        export_grid.addWidget(self.export_accuracy, 0, 1, 1, 1)
        export_grid.addWidget(self.filter_accuracy_btn, 0, 0, 1, 1)
        export_grid.addWidget(self.export_btn, 1, 0, 1, 2)

        self.grid.addWidget(view_box, 0, 0, -3, 1)
        self.grid.addWidget(predict_box, 0, 1, 1, 1)

        self.grid.addWidget(self.class_box, 2, 1, 1, 1)
        self.grid.addWidget(export_box, 3, 1, 1, 1)

        mainWidget = QtWidgets.QWidget()
        mainWidget.setLayout(self.grid)
        self.setCentralWidget(mainWidget)

    def predict(self):

        if 'self.locs' not in locals():
            msgBox = QtWidgets.QMessageBox(self)
            msgBox.setWindowTitle("Error")
            msgBox.setText("No localization data loaded.")
            msgBox.exec_()
        else:
            if (self.predicting is False) and (self.model_loaded is True):

                self.predicting = True

                self.oversampling = self.model_info["Oversampling"]
                self.pick_diameter = self.model_info["Pick Diameter"]
                self.pick_radius = self.pick_diameter / 2

                self.thread = Predicter(
                    self.model, self.locs, self.pick_radius, self.oversampling,
                )
                self.thread.predictions_made.connect(self.on_progress)

                self.thread.prediction_finished.connect(self.on_finished)
                self.thread.start()

    def on_finished(self, locs):
        self.locs = locs.copy()
        self.predicting = False

        self.status_bar.showMessage('Prediction finished.')

    def on_progress(self, pick, total_picks):
        # self.locs = locs.copy()
        self.status_bar.showMessage(
                                    "From {} picks - predicted {}".format(total_picks, pick)
        )

    def open(self):
        path, exe = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open localizations", filter="*.hdf5"
        )
        if path:
            self.open_file(path)

    def open_file(self, path):
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

            self.update_image()
            self.status_bar.showMessage("{} picks loaded. Ready for processing."
                                        .format(str(groups_max)))

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
            self.open_file(path)

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
            self.nanotron_log['Model Path'] = path
        except Exception:
            raise ValueError("No model file loaded.")

        try:
            with open(path[:-3]+'yaml', "r") as f:
                self.model_info = yaml.load(f, Loader=yaml.FullLoader)
                self.classes = []
                self.classes = self.model_info["Classes"]
                self.model_loaded = True
        except io.NoMetadataFileError:
            return

    def load_model(self):

        path, exe = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load model file", filter="*.sav", directory=None)
        if path:

            try:
                self.model = joblib.load(path)
                self.nanotron_log['Model Path'] = path
            except Exception:
                raise ValueError("No model file loaded.")

            try:
                with open(path[:-3]+'yaml', "r") as f:
                    self.model_info = yaml.load(f, Loader=yaml.FullLoader)
                    self.classes = []
                    self.classes = self.model_info["Classes"]
                    self.model_loaded = True
            except io.NoMetadataFileError:
                return

    def update_class_buttons(self):

        for id, name in self.classes.items():
            c = QtWidgets.QCheckBox(name)
            c.setChecked(True)
            self.classbox_grid.addWidget(c)

    def export(self):

        if 'self.locs' not in locals():
            msgBox = QtWidgets.QMessageBox(self)
            msgBox.setWindowTitle("Error")
            msgBox.setText("No localization data loaded.")
            msgBox.exec_()
            return

        if not hasattr(self.locs, "prediction"):
            msgBox = QtWidgets.QMessageBox(self)
            msgBox.setWindowTitle("Error")
            msgBox.setText("No predictions. Predict first.")
            msgBox.exec_()
            return

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
            if export_map[key] is True:
                export_classes[key] = item

        all_picks = len(np.unique(self.locs['group']))
        accuracy = self.export_accuracy.value()

        if self.filter_accuracy_btn.isChecked():
            print('Probability filter set to {:4}%'.format(accuracy*100))
            self.locs = self.locs[self.locs['score'] >= accuracy]
            dropped_picks = all_picks - len(np.unique(self.locs['group']))
            print("Dropped {} from {} picks.".format(dropped_picks, all_picks))
            self.nanotron_log['Probability'] = accuracy

        for prediction, name in export_classes.items():

            filtered_locs = self.locs[self.locs['prediction'] == prediction]
            n_groups = np.unique(filtered_locs['group'])
            n_new_groups = np.arange(0, len(n_groups), 1)
            regroup_dict = dict(zip(n_groups, n_new_groups))
            regroup_map = [regroup_dict[_] for _ in filtered_locs['group']]
            filtered_locs['group'] = regroup_map
            print('Regrouped datatset {} to {} picks.'.format(name, len(n_groups)))

            nanotron_info = self.nanotron_log.copy()
            nanotron_info.update({"Generated by": "Picasso Nanotron"})
            info = self.info + [nanotron_info]

            out_filename = '_' + name.replace(" ", "_").lower() + ".hdf5"
            out_path = os.path.splitext(self.path)[0] + out_filename
            io.save_locs(out_path, filtered_locs, info)

        print('Export of all predicted datasets finished.')
        self.status_bar.showMessage("{} files exported.".format(len(export_classes.items())))


def main():

    app = QtWidgets.QApplication(sys.argv)
    this_directory = os.path.dirname(os.path.realpath(__file__))
    icon_path = os.path.join(this_directory, "icons", "nanotron.ico")
    app.setWindowIcon(QIcon(icon_path))
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
