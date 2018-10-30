import numpy as  np
import matplotlib.pyplot as plt
import cv2 as cv
from skimage.morphology import skeletonize
from skimage import img_as_float
import argparse
import scipy.io as sio
import pickle
from pathlib import Path
import os

def load_ground_truth(path):
    mat = sio.loadmat(path)
    img = mat['groundTruth']['Segmentation']
    return  img[0,0]

def sample(img, grd_mask, pos_neg_ratio, h):
    #grd_mask = skeletonize(grd_mask > 1)
    pos_mask = (grd_mask > 1).astype(np.float32)
    pos_idxs = np.nonzero(pos_mask)
    n_pos_sample = len(pos_idxs[0])
    pos_samples = []
    pos_masks = []
    D0, D1 = img.shape[:2]
    for i in range(n_pos_sample):
        p0 = pos_idxs[0][i]
        p1 = pos_idxs[1][i]
        if p0-h < 0 or p0+h+1 >= D0 or p1-h < 0 or p1+h+1 >= D1:
            #print(x,p1, ': invalid')
            continue
        patch = img[p0 - h:p0 + h + 1, p1 - h:p1 + h + 1, :]
        patch = (patch.astype(np.float)/255.0)
        patch_msk = pos_mask[p0 - h:p0 + h + 1, p1 - h:p1 + h + 1]
        patch_msk = cv.resize(patch_msk, (5,5), cv.INTER_AREA)
        patch_msk = patch_msk.flatten()
        if (patch.shape != (27,27,3)):
            print(patch.shape)
            assert False

        if (patch_msk.shape != (25,)):
            print(patch_msk.shape)
            assert False

        pos_samples.append(patch)
        pos_masks.append(patch_msk)

    neg_mask = (grd_mask == 1).astype(np.uint8)
    neg_mask = cv.morphologyEx(neg_mask, cv.MORPH_ERODE, cv.getStructuringElement(cv.MORPH_RECT, (h,h)))
    neg_idxs = np.nonzero(neg_mask>0)
    n_neg_total_sample = len(neg_idxs[0])

    #just take a subset of negative samples
    n_neg_sample = min(pos_neg_ratio*n_pos_sample, n_neg_total_sample)

    neg_sample_idxs = np.random.choice(np.arange(n_neg_total_sample), n_neg_sample, replace=False)

    neg_samples = []
    neg_masks = []
    for idx in neg_sample_idxs:
        p0 = neg_idxs[0][idx]
        p1 = neg_idxs[1][idx]
        if p0-h < 0 or p0+h+1 >= D0 or p1-h < 0 or p1+h+1 >= D1:
            #print(x,p1, ': invalid')
            continue
        patch = img[p0-h:p0 + h+1, p1 - h:p1+h+1, :]
        if (patch.shape != (27,27,3)):
            assert False
            print(patch.shape)

        patch = (patch.astype(np.float)/255.0)
        neg_samples.append(patch)
        mask = np.zeros(shape=25, dtype=np.float)
        neg_masks.append(mask)

    return np.array(pos_samples), np.array(pos_masks), np.array(neg_samples), np.array(neg_masks)

def load_and_randomize_data(file_names, DATA_POS_DIR, MASK_POS_DIR, DATA_NEG_DIR, MASK_NEG_DIR):
    samples = []
    masks = []
    for name in file_names:
        dpath = f'{DATA_POS_DIR}{name}'
        cur_samples = np.load(dpath)
        cur_samples = [cur_samples[i,:] for i in range(cur_samples.shape[0])]

        mpath = f'{MASK_POS_DIR}{name}'
        cur_masks = np.load(mpath)
        cur_masks = [cur_masks[i,:] for i in range(cur_masks.shape[0])]

        dpath = f'{DATA_NEG_DIR}{name}'
        tmp_samples = np.load(dpath)
        cur_samples.extend(tmp_samples[i,:] for i in range(tmp_samples.shape[0]))

        mpath = f'{MASK_NEG_DIR}{name}'
        tmp_masks = np.load(mpath)
        cur_masks.extend(tmp_masks[i,:] for i in range(tmp_masks.shape[0]))

        rand_idxs = np.random.choice(np.arange(len(cur_samples)), len(cur_samples), replace=False).flatten()

        cur_samples = [cur_samples[i] for i in rand_idxs]
        cur_masks = [cur_masks[i] for i in rand_idxs]
        samples.extend(cur_samples)
        masks.extend(cur_masks)

    rand_idxs = np.random.choice(np.arange(len(samples)), len(samples), replace=False).flatten()

    return np.array([samples[i] for i in rand_idxs]), np.array([masks[i] for i in rand_idxs])

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="input data dir")
    ap.add_argument("-o", "--output", required=True, help="output data dir")
    ap.add_argument("-u", "--update", required=True, help="recalculate data")
    args = vars(ap.parse_args())

    update = int(args['update'])
    IN_DIR = args['input']
    OUT_DIR = args['output']

    IMG_DIR = f'{IN_DIR}/image/'
    GRD_DIR = f'{IN_DIR}/groundTruth/'
    DATA_POS_DIR = f'{OUT_DIR}/pos/'
    DATA_NEG_DIR = f'{OUT_DIR}/neg/'
    MASK_POS_DIR = f'{OUT_DIR}/pos_mask/'
    MASK_NEG_DIR = f'{OUT_DIR}/neg_mask/'

    if update > 0:
        os.makedirs(DATA_POS_DIR, exist_ok = True)
        os.makedirs(DATA_NEG_DIR, exist_ok = True)
        os.makedirs(MASK_POS_DIR, exist_ok = True)
        os.makedirs(MASK_NEG_DIR, exist_ok = True)

        for img_path in Path(IMG_DIR).glob('*.*'):
            truth_path = f'{GRD_DIR}{img_path.stem}.mat'
            if not os.path.isfile(truth_path):
                continue
            img = cv.imread(str(img_path))
            mask = load_ground_truth(truth_path)
            pos_samples, pos_masks, neg_samples, neg_masks = sample(img, mask, pos_neg_ratio=3, h=13)

            sample_path = f'{DATA_POS_DIR}{img_path.stem}'
            np.save(sample_path, pos_samples)

            sample_path = f'{MASK_POS_DIR}{img_path.stem}'
            np.save(sample_path, pos_masks)

            sample_path = f'{DATA_NEG_DIR}{img_path.stem}'
            np.save(sample_path, neg_samples)

            sample_path = f'{MASK_NEG_DIR}{img_path.stem}'
            np.save(sample_path, neg_masks)

            assert len(pos_samples) == len(pos_masks)
            print(f'{img_path.stem}: n_pos = {len(pos_samples)}, n_neg = {len(neg_samples)}')

    names = [fpath.name for fpath in Path(DATA_POS_DIR).glob('*.*')]

    split = int(0.7 * (len(names)))
    train_files = names[0:5]
    test_files = names[30:32]

    train_x, train_y = load_and_randomize_data(train_files, DATA_POS_DIR, MASK_POS_DIR, DATA_NEG_DIR, MASK_NEG_DIR)
    with open(f'{OUT_DIR}/train.pkl', mode='wb') as f:
        pickle.dump({'X':train_x, 'Y':train_y}, f)
    del train_x
    del train_y

    test_x, test_y = load_and_randomize_data(test_files, DATA_POS_DIR, MASK_POS_DIR, DATA_NEG_DIR, MASK_NEG_DIR)
    with open(f'{OUT_DIR}/test.pkl', mode='wb') as f:
        pickle.dump({'X':test_x, 'Y':test_y}, f)