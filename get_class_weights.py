import argparse
import sys
import os
import os.path as osp
import numpy as np
import cv2 as cv


# https://www.haya-programming.com/entry/2018/05/17/123000
def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--images_dir',
            help='images directory (png image in palette mode)', required=True)
    parser.add_argument('--classnames_file',
            help='classnames filename', required=True)
    args = parser.parse_args()

    if not osp.exists(args.images_dir):
        print('images_dir does not exist.')
        sys.exit(1)
    if not osp.exists(args.classnames_file):
        print('classnames file does not exist.')
        sys.exit(1)

    classes = []
    with open(args.classnames_file) as fh:
        classes = fh.read().split('\n')
    classes = classes[1:]  # remove 'ignore'
    n_classes = len(classes)

    counts = np.zeros((n_classes,), dtype=np.int64)
    for fi, f in enumerate(os.listdir(args.images_dir)):
        print(f'{fi}: {f}')
        img = cv.imread(osp.join(args.images_dir, f))
        bins = np.bincount(img.flatten()) // 3
        for i, bincount in enumerate(bins):
            counts[i] += bincount
    print('')
    print(f'classes: {classes}')
    print(f'bincounts: {counts}')
    n_samples = np.sum(counts)
    print(f'n_samples: {n_samples}')
    class_weights = n_samples / (n_classes * counts)
    print(f'class weights: {class_weights}')


if __name__ == '__main__':
    main()
