import cv2 as cv
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class MaskDrawer:
    def __init__(self, rect, img, mask):
        self.mask_axis = rect
        self.cidpress = self.mask_axis.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.mask_axis.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.mask_axis.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)
        self.cidscroll = self.mask_axis.figure.canvas.mpl_connect(
            'scroll_event', self.on_scroll)

        self.press = None
        self.radius = 3  # brush thickness
        self.img = img
        self.draw_img = img.copy()
        self.draw_value = (255,255,255)
        self.mask = mask
        self.cur_x = 0
        self.cur_y = 0
        self.LEFT_MOUSE_BUTTON = 1
        self.RIGHT_MOUSE_BUTTON = 3

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
        self.draw_img = cv.addWeighted(self.img, 0.5, self.mask, 0.5, gamma=0)

    def draw_mask(self, x, y):
        cv.circle(self.mask, (x,y), self.radius, self.draw_value, -1)
        return cv.addWeighted(self.img, 0.5, self.mask, 0.5, gamma=0)

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
    args = vars(ap.parse_args())
    IN_DIR = args['input']

    for path in Path(IN_DIR).glob('*.*'):
        img = cv.imread(str(path))
        img = cv.GaussianBlur(img, (5,5), 3)
        break

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('draw mask')
    mask_ax = ax.imshow(img)
    mask = np.zeros_like(img)
    linebuilder = MaskDrawer(mask_ax, img, mask)

    plt.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # rects = ax.bar(range(10), 20 * np.random.rand(10))
    # drs = []
    # for rect in rects:
    #     dr = DraggableRectangle(rect)
    #     dr.connect()
    #     drs.append(dr)
    # plt.show()
