import cv2 as cv
import numpy as np
from pathlib import Path

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

def extract_patch_1(img, h, offset, stride):
    patches = []
    patch_locs = []
    d = h+offset
    for i in range(d, img.shape[0]-d-1, stride):
        for j in range(d, img.shape[1]-d-1, stride):
            patch = img[i-h:i+h, j-h:j+h, :]
            if patch.shape != (2*h, 2*h, 3):
                print(patch.shape)
            patches.append(patch)
            patch_locs.append((i,j))
    return patches, patch_locs

def cut_image_into_patches(IN_DIR, OUT_DIR, img_target_size = (1024, 1024), patch_size = 224):
    win_size = 224
    h = int(win_size / 2)
    for path in Path(IN_DIR).glob("*.*"):
        img = cv.imread(str(path))
        patches, patch_locs = extract_patch_1(img, h, offset=0, stride=win_size)
        patches_1, patch_locs_1 = extract_patch_1(img, h, offset=h, stride=win_size)
        patches = patches + patches_1
        patch_locs = patch_locs + patch_locs_1
        for patch, loc in zip(patches, patch_locs):
            out_path = f'{OUT_DIR}/{path.stem}_{loc[0]}_{loc[1]}.png'
            cv.imwrite(out_path, patch)

def predict_image(model, img, crack_id = 0, debug_ax = None):
    win_size = 224
    h = int(win_size / 2)
    patches, patch_locs = extract_patch_1(img, h, offset=0, stride=win_size)
    patches_1, patch_locs_1 = extract_patch_1(img, h, offset=h, stride=win_size)
    patches = patches+patches_1
    patch_locs= patch_locs + patch_locs_1

    #do it due to out of memory
    pred_probs = []
    for patch in patches:
        input = np.expand_dims(patch, 0)
        prob = model.predict(input)
        pred_probs.append(prob.flatten())

    pred_probs = np.array(pred_probs)
    print(pred_probs)
    print(pred_probs.shape)
    preds = np.argmax(pred_probs, axis=1)
    has_crack = False
    if np.sum(preds == crack_id) > 0:
        has_crack = True

    if debug_ax is not None:
        for i, pred in enumerate(preds):
            if pred == crack_id:
                loc = patch_locs[i]
                pt0 = (loc[0]-h+5, loc[1]-h+5)
                pt1 = (loc[0]+h-5, loc[1]+h-5)
                if loc in patch_locs_1:
                    cv.rectangle(img, pt0, pt1, color=(255, 255, 0), thickness=3)
                else:
                    cv.rectangle(img, pt0, pt1, color=(0, 255, 255), thickness=3)
            debug_ax.imshow(img)

    return has_crack

def draw_crack_rectangle(img, crack_locs, axis, h):
    for loc in crack_locs:
        pt0 = (loc[0] - h + 5, loc[1] - h + 5)
        pt1 = (loc[0] + h - 5, loc[1] + h - 5)
        cv.rectangle(img, pt0, pt1, color=(255, 255, 0), thickness=3)

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

