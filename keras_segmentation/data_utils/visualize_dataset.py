import numpy as np
import cv2
import random

from .augmentation import augment_seg
from .data_loader import get_pairs_from_paths

random.seed(0)
class_colors = [(random.randint(0,255),random.randint(0,255),random.randint(0,255)) for _ in range(5000)]


def visualize_segmentation_dataset(images_path, segs_path, n_classes, do_augment=False, image_augmenter=None, reshuffle_interval=None):
    img_seg_pairs = get_pairs_from_paths(images_path, segs_path)
    colors = class_colors

    print("Press any key to navigate. ")
    cnt = 0
    for im_fn, seg_fn in img_seg_pairs:
        img = cv2.imread(im_fn)
        seg = cv2.imread(seg_fn)
        print("Found the following classes", np.unique(seg))

        seg_img = np.zeros_like(seg)

        if do_augment:
            if image_augmenter is not None:
                if reshuffle_interval is not None and cnt % reshuffle_interval == 0:
                    image_augmenter.shuffle()
                img, seg[:,:,0] = image_augmenter.augment_seg(img, seg[:,:,0])
            else:
                img, seg[:,:,0] = augment_seg(img, seg[:,:,0])

        for c in range(n_classes):
            seg_img[:,:,0] += ((seg[:,:,0] == c) * (colors[c][0])).astype('uint8')
            seg_img[:,:,1] += ((seg[:,:,0] == c) * (colors[c][1])).astype('uint8')
            seg_img[:,:,2] += ((seg[:,:,0] == c) * (colors[c][2])).astype('uint8')

        h, w, c = img.shape
        nh, nw = h, w
        if nw > 600:
            nw = 600
            nh = int(nw * h / w)
        img = cv2.resize(img, (nw, nh))
        seg_img = cv2.resize(seg_img, (nw, nh))
        cv2.imshow("img", img)
        cv2.imshow("seg_img", seg_img)
        cnt += 1
        if cv2.waitKey(0) == ord('q'):
            break
    cv2.destroyAllWindows()
