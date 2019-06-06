# import argparse
# from keras.models import load_model
import glob
import cv2
import numpy as np
import random
import os
from tqdm import tqdm
from .train import find_latest_checkpoint
from .data_utils.data_loader import get_image_arr, get_segmentation_arr
import json
from .models.config import IMAGE_ORDERING
from . import metrics
from .models import model_from_name

import six
import cv2 as cv

random.seed(0)
class_colors = [(random.randint(0,255),random.randint(0,255),random.randint(0,255)) for _ in range(5000)]


def model_from_checkpoint_path(checkpoints_path):
    assert (os.path.isfile(checkpoints_path+"_config.json")), "Checkpoint not found."
    model_config = json.loads(open(checkpoints_path+"_config.json", "r").read())
    latest_weights = find_latest_checkpoint(checkpoints_path)
    assert (latest_weights is not None), "Checkpoint not found."
    model = model_from_name[model_config['model_class']](model_config['n_classes'], input_height=model_config['input_height'], input_width=model_config['input_width'])
    print("loaded weights", latest_weights)
    model.load_weights(latest_weights)
    return model


def predict(model=None, inp=None, out_fname=None, checkpoints_path=None, legend_classnames=None, out_legend_fname=None):
    if model is None and (checkpoints_path is not None):
        model = model_from_checkpoint_path(checkpoints_path)

    assert(inp is not None)
    assert((type(inp) is np.ndarray) or isinstance(inp, six.string_types)), "Input should be the CV image or the input file name"

    if isinstance(inp, six.string_types):
        inp = cv2.imread(inp)

    assert len(inp.shape) == 3, "Image should be h,w,3 "
    original_h = inp.shape[0]
    original_w = inp.shape[1]

    output_width = model.output_width
    output_height = model.output_height
    input_width = model.input_width
    input_height = model.input_height
    n_classes = model.n_classes

    x = get_image_arr(inp, input_width, input_height, data_format=IMAGE_ORDERING)
    pr = model.predict(np.array([x]))[0]
    pr = pr.reshape((output_height, output_width, n_classes)).argmax(axis=2)

    if out_fname is not None:
        seg_img = np.zeros((output_height, output_width, 3))
        colors = class_colors
        for c in range(n_classes):
            seg_img[:,:,0] += ((pr[:,:] == c) * (colors[c][0])).astype('uint8')
            seg_img[:,:,1] += ((pr[:,:] == c) * (colors[c][1])).astype('uint8')
            seg_img[:,:,2] += ((pr[:,:] == c) * (colors[c][2])).astype('uint8')
        seg_img = cv2.resize(seg_img, (original_w, original_h))
        cv2.imwrite(out_fname, seg_img)

    if legend_classnames is not None and out_legend_fname is not None:
        pad = 10
        h = len(legend_classnames) * 12 + (len(legend_classnames) - 1) * 4
        w = 12 + 4 + 100
        legend_img = np.zeros((h, w, 3), dtype=np.uint8)
        legend_img = np.pad(legend_img, [(pad, pad), (pad, pad), (0, 0)], 'constant')
        for i, classname in enumerate(legend_classnames):
            color = colors[i]
            h_start = pad + i * (12 + 4)
            h_end = h_start + 12
            w_start = pad + 0
            w_end = w_start + 12
            legend_img[h_start:h_end, w_start:w_end] = color
            cv.putText(legend_img, classname, (w_end+8, h_end),
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
        cv2.imwrite(out_legend_fname, legend_img)

    return pr


def predict_multiple(model=None, inps=None, inp_dir=None, out_dir=None, checkpoints_path=None):
    if model is None and (checkpoints_path is not None):
        model = model_from_checkpoint_path(checkpoints_path)

    if inps is None and (inp_dir is not None):
        inps = glob.glob(os.path.join(inp_dir,"*.jpg")) + glob.glob(os.path.join(inp_dir,"*.png")) + glob.glob(os.path.join(inp_dir,"*.jpeg"))

    assert type(inps) is list
    # all_prs = []

    for i, inp in enumerate(tqdm(inps)):
        if out_dir is None:
            out_fname = None
        else:
            if isinstance(inp, six.string_types):
                out_fname = os.path.join(out_dir, os.path.basename(inp))
            else:
                out_fname = os.path.join(out_dir, str(i) + ".jpg")

        pr = predict(model, inp, out_fname)
        # all_prs.append(pr)
        yield inp, pr
    # return all_prs


def evaluate(model=None, inp_images=None, annotations=None, checkpoints_path=None):
    assert False, "not implemented "

    ious = []
    for inp, ann in tqdm(zip(inp_images, annotations)):
        pr = predict(model, inp)
        gt = get_segmentation_arr(ann, model.n_classes, model.output_width, model.output_height)
        gt = gt.argmax(-1)
        iou = metrics.get_iou(gt, pr, model.n_classes)
        ious.append(iou)
    ious = np.array(ious)
    print("Class wise IoU ", np.mean(ious, axis=0))
    print("Total  IoU ", np.mean(ious))
