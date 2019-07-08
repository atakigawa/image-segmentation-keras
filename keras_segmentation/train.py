import sys
import json
import os.path as osp
import os
import glob
import six
from .data_utils.data_loader import image_segmentation_generator2, verify_segmentation_dataset, get_pairs_from_paths
from .models import model_from_name
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from .weighted_categorical_crossentropy import weighted_categorical_crossentropy

import random
random.seed(0)


def find_latest_checkpoint(checkpoints_path):
    ckpts = glob.glob(osp.join(checkpoints_path, 'ep*.h5'))
    ckpts.sort()
    return ckpts[-1]


def train(model,
        train_images,
        train_annotations,
        input_height=None,
        input_width=None,
        n_classes=None,
        epochs=5,
        batch_size=2,
        verify_dataset=True,
        checkpoints_path=None,
        auto_resume_checkpoint=False,
        load_weights=None,
        optimizer='adadelta',
        loss_func='categorical_crossentropy',
        metrics=['accuracy'],
        validate=False,
        val_split=0.1,
        class_weight=None,
        do_augment_on_training=False,
        image_augmenter=None):

    if checkpoints_path is None:
        print('ERR: checkpoints_path is required.')
        sys.exit(1)

    if isinstance(model, six.string_types):  # check if user gives model name insteead of the model object
        # create the model from the name
        assert (n_classes is not None), "Please provide the n_classes"
        if (input_height is not None) and (input_width is not None):
            model = model_from_name[model](n_classes,
                    input_height=input_height, input_width=input_width)
        else:
            model = model_from_name[model](n_classes)

    n_classes = model.n_classes
    input_height = model.input_height
    input_width = model.input_width
    output_height = model.output_height
    output_width = model.output_width

    num_train = len(os.listdir(train_annotations))
    num_val = 0
    if validate:
        num_val = int(num_train * val_split)
        num_train = num_train - num_val

    if class_weight is not None:
        loss_func = weighted_categorical_crossentropy(class_weight)
    model.compile(loss=loss_func, optimizer=optimizer, metrics=metrics)

    os.makedirs(checkpoints_path, exist_ok=True)
    with open(osp.join(checkpoints_path, "info.json"), "w") as fh:
        fh.write(json.dumps({
            "model_class": model.model_name,
            "n_classes": n_classes,
            "input_height": input_height,
            "input_width": input_width,
            "output_height": output_height,
            "output_width": output_width
        }))

    log_dir = checkpoints_path
    if log_dir[-1] != '/':
        log_dir += '/'
    cb_monitor_target = 'loss'
    cb_ckpt_filename_tpl = 'ep{epoch:03d}-loss{loss:.3f}.h5'
    if validate:
        cb_monitor_target = 'val_loss'
        cb_ckpt_filename_tpl = 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'

    cb_logging = TensorBoard(log_dir=log_dir)
    cb_checkpoint = ModelCheckpoint(log_dir + cb_ckpt_filename_tpl,
            monitor=cb_monitor_target,
            save_weights_only=True,
            save_best_only=True,
            period=3)
    cb_early_stopping = EarlyStopping(
            monitor=cb_monitor_target,
            min_delta=0,
            patience=10,
            verbose=1)

    if (load_weights is not None) and len(load_weights) > 0:
        print("Loading weights from ", load_weights)
        model.load_weights(load_weights)

    if auto_resume_checkpoint:
        latest_checkpoint = find_latest_checkpoint(checkpoints_path)
        if latest_checkpoint is not None:
            print("Loading the weights from latest checkpoint ", latest_checkpoint)
            model.load_weights(latest_checkpoint)

    if verify_dataset:
        print("Verifying train dataset")
        verify_segmentation_dataset(train_images, train_annotations, n_classes)

    train_img_seg_pair = get_pairs_from_paths(train_images, train_annotations)
    random.shuffle(train_img_seg_pair)

    if validate:
        train_gen = image_segmentation_generator2(
                train_img_seg_pair[:num_train], batch_size, n_classes,
                input_height, input_width, output_height, output_width,
                do_augment=do_augment_on_training, image_augmenter=image_augmenter)
        val_gen = image_segmentation_generator2(
                train_img_seg_pair[num_train:], batch_size, n_classes,
                input_height, input_width, output_height, output_width)
        model.fit_generator(
                train_gen,
                steps_per_epoch=max(1, num_train // batch_size),
                validation_data=val_gen,
                validation_steps=max(1, num_val // batch_size),
                epochs=epochs,
                callbacks=[cb_logging, cb_checkpoint, cb_early_stopping])
    else:
        train_gen = image_segmentation_generator2(
                train_img_seg_pair, batch_size, n_classes,
                input_height, input_width, output_height, output_width,
                do_augment=do_augment_on_training, image_augmenter=image_augmenter)
        model.fit_generator(
                train_gen,
                steps_per_epoch=max(1, num_train // batch_size),
                epochs=epochs,
                callbacks=[cb_logging, cb_checkpoint, cb_early_stopping])

    final_weight_path = osp.join(checkpoints_path, 'weights_final.h5')
    model.save_weights(final_weight_path)
    print("saved ", final_weight_path)
