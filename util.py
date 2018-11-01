import cv2 as cv
import numpy as np

def extract_patch(img, h):
    patches = []
    patch_locs = []
    for i in range(h, img.shape[0]-h-1):
        for j in range(h, img.shape[1]-h-1):
            patch = img[i-h:i+h+1, j-h:j+h+1, :]
            if patch.shape != (2*h+1, 2*h+1, 3):
                print(patch.shape)
            patches.append(patch)
            patch_locs.append((i,j))
    return patches, patch_locs

def upsample_patch_predict(patch_pred, patch_locs, img_shape, h):
    img = np.zeros(img_shape[:2], dtype=np.float32)
    pred_size = int(np.sqrt(patch_pred.shape[1]))
    patch_pred = patch_pred.reshape(patch_pred.shape[0], pred_size, pred_size)
    for idx, loc in enumerate(patch_locs):
        i = loc[0]
        j = loc[1]
        pred = patch_pred[idx, :, :]
        pred = cv.resize(pred, dsize=(2*h+1, 2*h+1), interpolation=cv.INTER_AREA)
        img[i - h:i + h + 1, j - h:j + h + 1] = pred
    return img

