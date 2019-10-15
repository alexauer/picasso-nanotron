"""
    gui/nanotron
    ~~~~~~~~~~~~~~~~~~~~
    Graphical user interface for segmentation using deep learning
    :author: Alexander Auer, 2019
    :copyright: Copyright (c) 2016 Jungmann Lab, MPI of Biochemistry
"""
import os.path as _ospath
import os
import sys
import traceback
from tqdm import tqdm
import datetime

import matplotlib.pyplot as plt
import numba
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
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


class Generator(QtCore.QThread):

    datasets_made = QtCore.pyqtSignal(int, int, int, int)
    datasets_finished = QtCore.pyqtSignal(list, list)

    def __init__(self, locs, classes, pick_radius, oversampling, export, export_paths, parent=None):
        super().__init__()
        self.locs_files = locs.copy()
        self.pick_radius = pick_radius
        self.oversampling = oversampling
        self.classes = classes
        self.n_datasets = len(self.locs_files)
        self.export = export
        self.export_paths = export_paths
        print(self.n_datasets)

    def combine_data_sets(self, X_files, Y_files):

        # X = [img for xfile in X_files for img in xfile]
        # Y = [label for yfile in X_files for label in yfile]

        X = []
        Y = []
        for img in X_files:
            X += img

        for label in Y_files:
            Y += label

        return X, Y

    def run(self):

        X_files = []
        Y_files = []

        for id, locs in self.locs_files.items():

            img_shape = int(2 * self.pick_radius * self.oversampling)
            data = []
            labels = []
            label = self.classes[id]
            n_locs = locs.group.max()

            export_path = _ospath.dirname(self.export_paths[id]) + '/'

            for c, pick in enumerate(tqdm(np.unique(locs.group), desc='Prepare class ' + str(label))):

                pick_img = nanotron.roi_to_img(locs=locs,
                                               pick=pick,
                                               radius=self.pick_radius,
                                               oversampling=self.oversampling)

                if self.export is True and pick < 10:
                    filename = str(label).replace(" ", "_").lower() + '-' + str(pick)
                    plt.imsave(export_path + filename + '.png', (10*pick_img-1),
                               cmap='Greys')

                pick_img = nanotron.prepare_img(pick_img,
                                                img_shape=img_shape,
                                                alpha=10,
                                                bg=1)

                self.datasets_made.emit(id+1, self.n_datasets, c+1, n_locs+1)
                data.append(pick_img)
                labels.append(id)

            X_files.append(data)
            Y_files.append(labels)

        X_train, Y_train = self.combine_data_sets(X_files, Y_files)

        self.datasets_finished.emit(X_train, Y_train)


class Trainer(QtCore.QThread):

    training_finished = QtCore.pyqtSignal(list, float, float, list)

    def __init__(self, X_train, Y_train, parameter, parent=None):
        super().__init__()
        self.nodes = parameter["nodes"]
        self.activation = parameter["activation"]
        self.iterations = parameter["iterations"]
        self.learning_rate = parameter["learning_rate"]
        self.solver = parameter["solver"]
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X_train,
                                                                                Y_train,
                                                                                test_size=0.30,
                                                                                random_state=42)
        self.mlp_list = []  # container to carry mlp class
        self.cm_list = []

    def run(self):
        self.mlp_list = []
        mlp = MLPClassifier(hidden_layer_sizes=(self.nodes,),
                            activation=self.activation,
                            max_iter=self.iterations,
                            alpha=0.01,
                            solver=self.solver,
                            verbose=False,
                            shuffle=True,
                            tol=1e-4,
                            random_state=1,
                            validation_fraction=0.15,
                            learning_rate_init=self.learning_rate)

        mlp.fit(self.X_train, self.Y_train)
        score = mlp.score(self.X_train, self.Y_train)
        test_score = mlp.score(self.X_test, self.Y_test)
        self.mlp_list.append(mlp)

        Y_pred = mlp.predict(self.X_test)
        cm = confusion_matrix(self.Y_test, Y_pred)
        self.cm_list.append(cm)

        self.training_finished.emit(self.mlp_list, score, test_score, self.cm_list)


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

        self.prediction = np.zeros(len(np.unique(self.locs['group'])),
                                   dtype=[('group', 'u4'),
                                   ('prediction', 'i4'), ('score', 'f4')])
        self.prediction['group'] = np.unique(self.locs['group'])
        n_groups = len(np.unique(self.locs['group']))
        p_locs = np.zeros(len(self.locs['group']), dtype=[('group', 'u4'),
                          ('prediction', 'i4'), ('score', 'f4')])

        for id, pick in enumerate(tqdm(self.prediction['group'],
                                  desc='Predict')):

            self.predictions_made.emit(pick, n_groups)

            pred, pred_proba = nanotron.predict_structure(mlp=self.model,
                                                          locs=self.locs,
                                                          pick=pick,
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
        self.training_files = {}
        self.training_files_path = {}
        self.classes = {}
        self.classes_name = []
        self.f_btns = []
        self.pick_radii = {}
        self.X_train = []
        self.Y_train = []
        self.generator_running = False
        self.trainer_running = False
        self.train_log = {}

        choose_n_files_box = QtWidgets.QGroupBox("Number of Classes")
        choose_class_grid = QtWidgets.QGridLayout(choose_n_files_box)
        self.choose_n_files = QtWidgets.QSpinBox()
        self.choose_n_files.setRange(1, 6)
        self.choose_n_files.setValue(0)
        self.choose_n_files.setKeyboardTracking(False)

        file_slots_btn = QtWidgets.QPushButton("Generate Data Set")
        file_slots_btn.clicked.connect(self.update_train_files)
        choose_class_grid.addWidget(QtWidgets.QLabel("Classes:"), 0, 0)
        choose_class_grid.addWidget(self.choose_n_files, 0, 1)
        choose_class_grid.addWidget(file_slots_btn, 0, 2)

        train_files_box = QtWidgets.QGroupBox("Training Files")
        self.train_files_grid = QtWidgets.QGridLayout(train_files_box)

        train_img_box = QtWidgets.QGroupBox("Image Parameter")
        self.train_img_grid = QtWidgets.QHBoxLayout(train_img_box)
        self.oversampling_box = QtWidgets.QSpinBox()
        self.oversampling_box.setRange(1, 200)
        self.oversampling_box.setValue(50)
        self.export_img = QtWidgets.QCheckBox("Export Image Subset")
        self.export_img.setChecked(False)

        self.train_img_grid.addWidget(QtWidgets.QLabel("Oversampling:"), 0)
        self.train_img_grid.addWidget(self.oversampling_box, 1)
        self.train_img_grid.addWidget(self.export_img, 2)

        prepare_data_btn = QtWidgets.QPushButton("Prepare Data")
        prepare_data_btn.clicked.connect(self.prepare_data)

        perceptron_box = QtWidgets.QGroupBox("Perceptron")
        perceptron_grid = QtWidgets.QGridLayout(perceptron_box)
        perceptron_grid.addWidget(QtWidgets.QLabel("Nodes:"), 0, 0)

        self.nodes = QtWidgets.QSpinBox()
        self.nodes.setRange(1, 1000)
        self.nodes.setValue(100)
        self.nodes.setKeyboardTracking(False)
        perceptron_grid.addWidget(self.nodes, 0, 1)

        perceptron_grid.addWidget(QtWidgets.QLabel("Solver:"), 2, 0)
        self.solver = QtWidgets.QComboBox()
        self.solver.addItems(['adam', 'lbfgs', 'sgd'])
        perceptron_grid.addWidget(self.solver, 2, 1)

        perceptron_grid.addWidget(QtWidgets.QLabel("Activation:"), 3, 0)
        self.activation_ft = QtWidgets.QComboBox()
        self.activation_ft.addItems(['relu', 'identity', 'logistic', 'tanh'])
        perceptron_grid.addWidget(self.activation_ft, 3, 1)

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

        self.train_btn = QtWidgets.QPushButton("Train Model")
        self.train_btn.clicked.connect(self.train)
        self.train_btn.setDisabled(True)
        self.train_label = QtWidgets.QLabel("")
        self.train_label.setAlignment(QtCore.Qt.AlignCenter)
        # train_btn.setVisible(False)
        self.learning_curve_btn = QtWidgets.QPushButton("Show Learning Curve")
        self.learning_curve_btn.clicked.connect(self.show_learning_stats)
        self.learning_curve_btn.setVisible(False)

        self.save_model_btn = QtWidgets.QPushButton("Save Model")
        self.save_model_btn.clicked.connect(self.save_model)
        self.save_model_btn.setVisible(False)

        self.score_box = QtWidgets.QGroupBox()
        score_grid = QtWidgets.QGridLayout(self.score_box)
        self.score_label_0_0 = QtWidgets.QLabel()
        self.score_label_0_1 = QtWidgets.QLabel("Acc")
        self.score_label_0_2 = QtWidgets.QLabel("Loss")
        self.score_label_1_0 = QtWidgets.QLabel("Train:")
        self.score_label_1_1 = QtWidgets.QLabel()
        self.score_label_1_2 = QtWidgets.QLabel()
        self.score_label_2_0 = QtWidgets.QLabel("Test:")
        self.score_label_2_1 = QtWidgets.QLabel()
        self.score_label_2_2 = QtWidgets.QLabel()
        score_grid.addWidget(self.score_label_0_0, 0, 0, 1, 1)
        score_grid.addWidget(self.score_label_0_1, 0, 1, 1, 1)
        score_grid.addWidget(self.score_label_0_2, 0, 2, 1, 1)
        score_grid.addWidget(self.score_label_1_0, 1, 0, 1, 1)
        score_grid.addWidget(self.score_label_1_1, 1, 1, 1, 1)
        score_grid.addWidget(self.score_label_1_2, 1, 2, 1, 1)
        score_grid.addWidget(self.score_label_2_0, 2, 0, 1, 1)
        score_grid.addWidget(self.score_label_2_1, 2, 1, 1, 1)
        score_grid.addWidget(self.score_label_2_2, 2, 2, 1, 1)
        self.score_box.setVisible(False)

        progress_box = QtWidgets.QGroupBox()
        progress_grid = QtWidgets.QGridLayout(progress_box)
        # progress_grid =
        self.progress_sets_label = QtWidgets.QLabel()
        self.progress_sets_label.setAlignment(QtCore.Qt.AlignLeft)
        self.progress_imgs_label = QtWidgets.QLabel()
        self.progress_imgs_label.setAlignment(QtCore.Qt.AlignRight)
        self.progress_bar = QtWidgets.QProgressBar(self)

        progress_grid.addWidget(self.progress_sets_label, 0, 0, 1, 1)
        progress_grid.addWidget(self.progress_imgs_label, 0, 1, 1, 1)
        progress_grid.addWidget(self.progress_bar, 1, 0, 1, 2)

        grid.addWidget(choose_n_files_box, 0, 0, 1, 1)
        grid.addWidget(train_files_box, 1, 0, 6, 1)
        grid.addWidget(train_img_box, 7, 0, 1, 1)
        grid.addWidget(prepare_data_btn, 8, 0, 1, 1)
        grid.addWidget(progress_box, 9, 0, 2, 1)

        grid.addWidget(perceptron_box, 0, 1, 3, 1)
        grid.addWidget(train_parameter_box, 3, 1, 1, 1)
        grid.addWidget(self.train_btn, 4, 1, 1, 1)
        grid.addWidget(self.learning_curve_btn, 5, 1, 1, 1)
        grid.addWidget(self.train_label, 7, 1, 1, 1)
        grid.addWidget(self.save_model_btn, 8, 1, 1, 1)
        grid.addWidget(self.score_box, 9, 1, 2, 1)


    def save_model(self):

        if self.mlp is not None:

            fname, ext = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save mode file",
                "model.sav",
                ".sav",
            )

            self.train_log["Model"] = fname
            self.train_log["Generated by"] = 'Picasso nanoTRON : Train'
            import sklearn
            self.train_log["Scikit-Learn Version"] = sklearn.__version__
            self.train_log["Created on"] = datetime.datetime.now()

            if fname:
                joblib.dump(self.mlp, fname)
                print("Saving complete.")
                base, ext = _ospath.splitext(fname)
                info_path = base + ".yaml"
                io.save_info(info_path, [self.train_log])

    def train(self):

        if self.data_prepared and not self.trainer_running:
            self.train_label.setText("Model is training...")
            self.score_label_1_1.setText('-')
            self.score_label_2_1.setText('-')
            self.score_label_1_2.setText('-')
            self.train_log["Classes"] = self.classes
            self.train_log["Training Files"] = self.training_files_path
            parameter = {}
            parameter["nodes"] = self.nodes.value()
            parameter["activation"] = self.activation_ft.currentText()
            parameter["iterations"] = self.iterations.value()
            parameter["learning_rate"] = self.learing_rate.value()
            parameter["solver"] = self.solver.currentText()
            self.train_thread = Trainer(X_train=self.X_train,
                                        Y_train=self.Y_train,
                                        parameter=parameter)

            self.train_thread.training_finished.connect(self.train_finished)
            self.train_thread.start()
            self.trainer_running = True
            print("Training started.")

    def show_learning_stats(self):
        if self.mlp is not None:

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax1.set_title("Learning Curve")
            ax1.plot(self.mlp.loss_curve_, label="Train")
            ax1.legend(loc="best")
            ax1.set_xlabel("Iterations")
            ax1.set_ylabel("Loss")

            im = ax2.imshow(self.cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax2.figure.colorbar(im, ax=ax2)
            ax2.set(xticks=np.arange(self.cm.shape[1]),
                    yticks=np.arange(self.cm.shape[0]),
                    xticklabels=self.classes.values(),
                    yticklabels=self.classes.values(),
                    title='Confusion Matrix',
                    ylabel='True label',
                    xlabel='Predicted label')

            plt.setp(ax2.get_yticklabels(), rotation='vertical',
                     horizontalalignment='right',
                     verticalalignment='center')

            thresh = self.cm.max() / 2.
            for i in range(self.cm.shape[0]):
                for j in range(self.cm.shape[1]):
                    ax2.text(j, i, format(self.cm[i, j], 'd'),
                             ha="center", va="center",
                             color="white" if self.cm[i, j] > thresh else "black")
            plt.autoscale()
            fig.show()

    def update_train_files(self):

        if not self.file_slots_generated:
            self.file_slots_generated = True

            for file in range(self.choose_n_files.value()):

                c = QtWidgets.QLabel('{}:'.format(file))
                self.train_files_grid.addWidget(c, file, 0)

                self.f_btn = QtWidgets.QPushButton("Load File", self)
                self.f_btn.clicked.connect(lambda _, fi=file: self.load_train_file(fi))
                self.train_files_grid.addWidget(self.f_btn, file, 1)
                self.f_btns.append(self.f_btn)

                la = QtWidgets.QLabel("Name:".format(file))
                self.train_files_grid.addWidget(la, file, 2)

                id = QtWidgets.QLineEdit(self)
                id.move(20, 20)
                id.resize(500, 40)
                id.setMaxLength(10)
                self.train_files_grid.addWidget(id, file, 3)
                self.classes_name.append(id)

    def load_train_file(self, file):

        path, exe = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open training localizations", filter="*.hdf5"
        )
        if path:
            try:
                locs, info = io.load_locs(path, qt_parent=self)
                self.pick_radii[file] = self.parse_pick_radius(info)
                self.training_files_path[file] = path
            except io.NoMetadataFileError:
                return

        if not hasattr(locs, "group"):
            msgBox = QtWidgets.QMessageBox(self)
            msgBox.setWindowTitle("Error")
            msgBox.setText(
                ("Datafile does not contain group information."
                 "Please load file with picked localizations.")
            )
            msgBox.exec_()
        else:
            self.training_files[(file)] = locs
            self.f_btns[file].setText("Loaded")
            self.f_btns[file].setEnabled(False)

    def parse_pick_radius(self, info):

        for num, file in enumerate(info):
            if "Pick Diameter" in file:
                diameter = info[num]["Pick Diameter"]
                radius = diameter / 2
        return radius

    def check_set(self):

        n_files = self.choose_n_files.value()
        n_datasets = len(self.training_files)

        passed = False
        n_ids = 0  # not very elegant

        for id, txt in enumerate(self.classes_name):
            name = txt.text().strip()
            if name:
                self.classes[id] = name
                n_ids += 1

        if (n_files == n_ids) and (n_files == n_datasets):
            passed = True

        return passed

    def prepare_data(self):

        if self.generator_running:
            msgBox = QtWidgets.QMessageBox(self)
            msgBox.setWindowTitle("Error")
            msgBox.setText("Preparation already running.")
            msgBox.exec_()
        elif (self.check_set()):
            self.generator_running = True
            # Get the largest pick radius
            max_key = max(self.pick_radii, key=lambda x: self.pick_radii.get(x))
            self.pick_radius = self.pick_radii[max_key]
            self.oversampling = self.oversampling_box.value()

            self.train_log["Pick Diameter"] = 2 * self.pick_radius
            self.train_log["Oversampling"] = self.oversampling

            self.generate_thread = Generator(
                                        locs=self.training_files,
                                        classes=self.classes,
                                        pick_radius=self.pick_radius,
                                        oversampling=self.oversampling,
                                        export=True,
                                        export_paths=self.training_files_path
                                         )
            self.generate_thread.datasets_made.connect(self.prepare_progress)
            self.generate_thread.datasets_finished.connect(self.prepare_finished)
            self.generate_thread.start()

        else:
            msgBox = QtWidgets.QMessageBox(self)
            msgBox.setWindowTitle("Error")
            msgBox.setText("No all data sets loaded or names defined.")
            msgBox.exec_()

    def prepare_progress(self, current_dataset, last_dataset, current_img, last_img):

        self.progress_sets_label.setText("{}/{}".format(current_dataset,
                                                        last_dataset))
        self.progress_imgs_label.setText("{}/{}".format(current_img,
                                                        last_img))
        self.progress_bar.setMaximum(last_img)
        self.progress_bar.setValue(current_img)

    def prepare_finished(self, X_train, Y_train):
        print("Training data generated.")
        self.train_btn.setDisabled(False)
        self.data_prepared = True
        self.X_train = X_train
        self.Y_train = Y_train

    def train_finished(self, mlp, score, test_score, cm):
        print("Training finished.")
        self.trainer_running = False
        self.mlp = mlp[0]
        self.cm = cm[0]
        self.train_label.setText("Training finished.")
        self.learning_curve_btn.setVisible(True)
        self.save_model_btn.setVisible(True)
        self.train_log["Train Accuracy"] = float(score)
        self.train_log["Test Accuracy"] = float(test_score)
        self.train_log["Train Loss"] = float(self.mlp.loss_)
        self.score_label_1_1.setText(("{:3.2f}").format(score))
        self.score_label_2_1.setText(("{:3.2f}").format(test_score))
        self.score_label_1_2.setText("{:.2e}".format(self.mlp.loss_))
        self.score_box.setVisible(True)


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
        self.setWindowTitle("Picasso: nanoTRON")
        self.resize(768, 512)
        this_directory = os.path.dirname(os.path.realpath(__file__))
        self.icon_path = os.path.join(this_directory, "icons", "nanotron.ico")
        icon = QtGui.QIcon(self.icon_path)
        self.setWindowIcon(icon)
        self.setAcceptDrops(True)
        self.predicting = False
        self.model_loaded = False
        self.nanotron_log = {}
        self.classes = []

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
        train_model_action.setShortcut("Ctrl+T")
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
        self.predict_btn.setDisabled(True)
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
        self.export_btn.setDisabled(True)

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
        self.export_btn.setDisabled(False)
        self.status_bar.showMessage('Prediction finished.')

    def on_progress(self, pick, total_picks):
        # self.locs = locs.copy()
        self.status_bar.showMessage("Predicting... Please wait. From {} picks - predicted {}"
                                    .format(total_picks, pick))

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
            self.predict_btn.setDisabled(False)
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
            print("Defaul model loaded.")
        except Exception:
            print("Default model not loaded.")
            self.status_bar.showMessage("Load model.")
            # raise ValueError("Default model not loaded.")

        try:
            with open(path[:-3]+'yaml', "r") as f:
                self.model_info = yaml.load(f)
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
                    self.model_info = yaml.load(f)
                    self.classes = []
                    self.classes = self.model_info["Classes"]
                    self.model_loaded = True
                    self.update_class_buttons()
            except io.NoMetadataFileError:
                return

    def clearLayout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()


    def update_class_buttons(self):

        self.clearLayout(self.classbox_grid)
        for id, name in self.classes.items():

            c = QtWidgets.QCheckBox(name)
            c.setChecked(True)
            self.classbox_grid.addWidget(c)

        self.classbox_grid.addStretch(1)

    def export(self):

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

        export_locs = self.locs.copy()

        if self.filter_accuracy_btn.isChecked():
            print('Probability filter set to {:4}%'.format(accuracy*100))
            export_locs = export_locs[export_locs['score'] >= accuracy]
            dropped_picks = all_picks - len(np.unique(export_locs['group']))
            print("Dropped {} from {} picks.".format(dropped_picks, all_picks))
            self.nanotron_log['Probability'] = accuracy

        for prediction, name in export_classes.items():

            filtered_locs = export_locs[export_locs['prediction'] == prediction]
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
