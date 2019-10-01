"""
    picasso.nanotron
    ~~~~~~~~~~

    Machine learning library for segmentation of picks

    :author: Alexander Auer 2019
    :copyright: Copyright (c) 2019 Jungmann Lab, MPI of Biochemistry
"""

import numpy as np
from tqdm import tqdm as tqdm

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from . import io, render, lib

def prepare_img(img, img_shape, alpha=1, bg=0):

    img = alpha * img - bg
    img = img.astype('float')
    img = img / img.max()
    img = img.clip(min=0)
    img = img.reshape(img_shape**2)

    return img

def roi_to_img(locs, pick, radius, oversampling):

    # Isolate locs from pick
    pick_locs = []
    pick_locs = locs[(locs.group == pick)]

    radius -=0.001 #dirty method to avoid floating point errors with render

    # Calculate viewport
    x_min = np.mean(pick_locs.x) - radius
    x_max = np.mean(pick_locs.x) + radius
    y_min = np.mean(pick_locs.y) - radius
    y_max = np.mean(pick_locs.y) + radius

    viewport =  (y_min, x_min), (y_max, x_max)

    if False: #for debugging
        print("mean x: {}".format(np.mean(pick_locs.x)))
        print('length x: {}'.format(x_max - x_min))
        print("mean y: {}".format(np.mean(pick_locs.y)))
        print('length y: {}'.format(y_max - y_min))
        print('radius: {}'.format(radius))
        print('viewport: {}'.format(viewport))

    # Render locs with Picasso render function
    len_x, pick_img = render.render(pick_locs, viewport = viewport, oversampling=oversampling, blur_method='smooth')

    return pick_img

def prepare_data(locs, identification, picks_radius, oversampling, img_shape, alpha=10, bg=1, export=False):

    data = []
    label = []

    for pick in tqdm(range(locs.group.max()), desc='Prepare class '+str(identification)):

        pick_img = roi_to_img(locs, pick, radius=pick_radius, oversampling=oversampling)

        if export == True and pick < 10:
            filename = 'id' + str(identification) + '-' + str(pick)
            plt.imsave('./img/' + filename + '.png', (alpha*pick_img-bg), cmap='Greys', vmax=10)

        pick_img = prepare_img(pick_img, img_shape=img_shape, alpha=alpha, bg=bg)

        data.append(pick_img)
        label.append(identification)

    return data, label

# def combine_data_to_4d(x_1, y_1, x_2, y_2 ):
#
#     x = x_1 + x_2
#     y = y_1 + y_2
#
#     return np.asarray(x), np.asarray(y)

def predict_structure(mlp, locs, pick, img_shape, pick_radius, oversampling):

    # Iterate through groups, render, predict and save append to according DataFrame

    img = roi_to_img(locs, pick=pick, radius=pick_radius, oversampling=oversampling)
    img = prepare_img(img, img_shape=img_shape, alpha=10, bg=1)
    img = img.reshape(1, img_shape**2)

    pred = mlp.predict(img)
    pred_proba = mlp.predict_proba(img)

    return pred, pred_proba
