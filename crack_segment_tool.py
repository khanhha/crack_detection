import cv2 as cv
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import skimage.feature as feature

class MaskPainter:
    def __init__(self, fig, mask_axis, result_axis, in_dir, out_dir):
        self.figure = fig
        self.mask_axis = mask_axis
        self.result_axis = result_axis
        self.cidpress = self.mask_axis.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.mask_axis.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.mask_axis.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cidscroll = self.mask_axis.figure.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.cidkeypress = self.mask_axis.figure.canvas.mpl_connect('key_press_event', self.on_key_press)

        self.IN_DIR = in_dir
        self.OUT_DIR = out_dir
        self.in_paths  = [path for path in Path(self.IN_DIR).glob('*.*')]

        self.press = None
        self.radius = 4  # brush thickness
        self.draw_value = (255,255,255)
        self.cur_x = 0
        self.cur_y = 0
        self.LEFT_MOUSE_BUTTON = 1
        self.RIGHT_MOUSE_BUTTON = 3

        self.initialize_img()

    def load_img(self, path):
        img = cv.imread(str(path))
        img = cv.resize(img, (540, 360))
        return img

    def initialize_img(self):
        img_path = None
        #find the first input path which hasn't been segmented yet
        out_paths = [path for path in Path(self.OUT_DIR).glob('*.*')]
        for ipath in self.in_paths:
            for opath in out_paths:
                if ipath.stem != opath.stem:
                    img_path = opath

        if img_path == None:
            img_path = self.in_paths[np.random.randint(0, len(self.in_paths))]

        self.set_up_new_img(img_path)
        self.mask_all = np.zeros((self.img.shape[0], self.img.shape[1], 3), dtype=np.uint8)

    def set_up_new_img(self, path):
        self.img = self.load_img(path)
        self.draw_img = self.img.copy()
        self.apply_drawing()

        opath = f'{OUT_DIR}/{path.name}'
        if Path(opath).exists():
            cur_segment = self.load_img(opath)
            self.result_axis.set_data(cur_segment)
            self.result_axis.figure.canvas.draw()

    def on_key_press(self, event):
        if event.key == 'c':
            print('start segmenting crack')
            bgdmodel = np.zeros((1, 65), np.float64)
            fgdmodel = np.zeros((1, 65), np.float64)

            mask = self.build_mask()
            cv.grabCut(self.img, mask, None, bgdmodel, fgdmodel, 1, cv.GC_INIT_WITH_MASK)

            mask_out = np.where((mask == cv.GC_FGD) + (mask == cv.GC_PR_FGD), 255, 0).astype('uint8')
            output = cv.bitwise_and(self.img, self.img, mask=mask_out)
            self.result_axis.set_data(output)
            self.result_axis.figure.canvas.draw()

        elif event.key == 'v':
            mask = self.mask_all == (255, 255, 255)
            mask = np.bitwise_and(np.bitwise_and(mask[:,:,0], mask[:,:,1]), mask[:,:,2])
            smooth_img = cv.GaussianBlur(self.img, ksize=(3,3), sigmaX=1)
            gray = cv.cvtColor(smooth_img, cv.COLOR_RGB2GRAY)
            output = self.auto_canny(gray, mask)
            self.mask_axis.set_data(smooth_img)
            self.mask_axis.figure.canvas.draw()
            self.result_axis.set_data(np.dstack([output, output, output]))
            self.result_axis.figure.canvas.draw()

    def auto_canny(self, image, mask, sigma=0.33):
        # compute the median of the single channel pixel intensities
        v = np.median(image[mask])
        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        print(f'median = {v}, lower = {lower}, upper = {upper}')
        edged = cv.Canny(image,lower, upper)
        # return the edged image
        edged[np.bitwise_not(mask)] = 0
        return edged

    def build_mask(self):
        mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
        mask[:] = cv.GC_BGD
        tmp = self.mask_all == (255, 255, 255)
        tmp = np.bitwise_and(np.bitwise_and(tmp[:,:,0], tmp[:,:,1]), tmp[:,:,2])
        mask[tmp] = cv.GC_PR_FGD
        return mask

    def on_press(self, event):
        #just process left and right button
        if event.button != self.LEFT_MOUSE_BUTTON and event.button != self.RIGHT_MOUSE_BUTTON:
            return

        self.draw_value = (255,255,255) if event.button == self.LEFT_MOUSE_BUTTON else (0, 0, 0)

        if event.inaxes != self.mask_axis.axes: return

        contains, attrd = self.mask_axis.contains(event)
        if not contains: return
        self.cur_x, self.cur_y = int(event.xdata), int(event.ydata)
        self.press = True
        self.draw_mask(self.cur_x, self.cur_y)
        self.apply_drawing()

    def on_motion(self, event):
        if event.inaxes != self.mask_axis.axes: return
        self.cur_x, self.cur_y = int(event.xdata), int(event.ydata)
        if self.press is not None:
            self.draw_img = self.draw_mask(self.cur_x, self.cur_y)
        else:
            self.reset_img()

        self.apply_drawing()

    def apply_drawing(self):
        cv.circle(self.draw_img, (self.cur_x, self.cur_y), radius = self.radius, color=(255, 0, 0), thickness=1, lineType=0)
        self.mask_axis.set_data(self.draw_img)
        self.mask_axis.figure.canvas.draw()

    def reset_img(self):
        self.draw_img = cv.addWeighted(self.img, 0.5, self.mask_all, 0.5, gamma=0)

    def draw_mask(self, x, y):
        cv.circle(self.mask_all, (x, y), self.radius, self.draw_value, -1)
        return cv.addWeighted(self.img, 0.5, self.mask_all, 0.5, gamma=0)

    def on_release(self, event):
        self.press = None
        self.mask_axis.figure.canvas.draw()

    def on_scroll(self, event):
        if event.button == 'up':
            delta = 1
        elif event.button == 'down':
            delta = -1
        else:
            delta = 0

        self.update_radius(delta)

    def update_radius(self, delta):
        self.radius += delta
        self.reset_img()
        self.apply_drawing()

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.mask_axis.figure.canvas.mpl_disconnect(self.cidpress)
        self.mask_axis.figure.canvas.mpl_disconnect(self.cidrelease)
        self.mask_axis.figure.canvas.mpl_disconnect(self.cidmotion)
        self.mask_axis.figure.canvas.mpl_disconnect(self.cidscroll)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="input data dir")
    ap.add_argument("-o", "--output", required=True, help="output data dir")

    args = vars(ap.parse_args())
    IN_DIR = args['input']
    OUT_DIR = args['output']

    in_paths  = [path for path in Path(IN_DIR).glob('*.*')]
    if len(in_paths) > 0:
        dummy_img = np.zeros((360, 540), dtype=np.uint8)

        fig = plt.figure()
        left_axis = fig.add_subplot(121)
        left_axis.set_title('draw mask')
        mask_ax = left_axis.imshow(dummy_img)
        mask = np.zeros_like(dummy_img)

        right_axis = fig.add_subplot(122)
        right_axis.set_title('crack segmentation result')
        result_ax = right_axis.imshow(dummy_img)

        painter = MaskPainter(fig, mask_ax, result_ax, in_dir=IN_DIR, out_dir=OUT_DIR)
        plt.show()
    else:
        print(f'no file in the input directory {IN_DIR}')