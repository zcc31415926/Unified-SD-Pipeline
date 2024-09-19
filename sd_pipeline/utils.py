import cv2
import numpy as np
from PIL import Image


class CenterPad:
    def __init__(self, pad_value):
        self.pad_value = pad_value

    def __call__(self, img):
        img_ = np.array(img)
        h, w, c = img_.shape
        canvas = np.ones((max(h, w), max(h, w), c)) * self.pad_value
        if h >= w:
            canvas[:, (h - w) // 2 : (h - w) // 2 + w] = img_
        else:
            canvas[(w - h) // 2 : (w - h) // 2 + h] = img_
        return Image.fromarray(canvas.astype(np.uint8))


def removeCenterPad(img, ori_size):
    W, H = ori_size
    h, w, _ = img.shape
    assert h == w, 'Images after transformation should be square'
    if H > W:
        pad_len = int((w - h * W / H) / 2)
        img = img[:, pad_len : pad_len + int(h * W / H)]
    else:
        pad_len = int((h - w * H / W) / 2)
        img = img[pad_len : pad_len + int(w * H / W)]
    return cv2.resize(img, (W, H))

