# Author: XC 2017

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from PIL import Image
import time
import argparse
import pyflow
import cv2 
import os 

parser = argparse.ArgumentParser(
    description='Demo for python wrapper of Coarse2Fine Optical Flow')
parser.add_argument(
    '-viz', dest='viz', action='store_true',
    help='Visualize (i.e. save) output of flow.')

parser.add_argument('--vid_src', dest='vid_src')
parser.add_argument('--of_dir', dest='of_dir')
parser.add_argument('--img_dir', dest='img_dir')
parser.add_argument('--crop', dest='crop', default=None, 
                        help='Input left,up,right,bottom')

args = parser.parse_args()

###################### Flow Options
alpha = 0.012
ratio = 0.75
minWidth = 20
nOuterFPIterations = 2
nInnerFPIterations = 1
nSORIterations = 10
colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))


##################### Video options 
DST_SIZE = (320, 240)


def save_of(im1, im2, fname):
    # im1 = np.array(Image.open('examples/car1.jpg'))
    # im2 = np.array(Image.open('examples/car2.jpg'))
    im1 = im1.astype(float) / 255.
    im2 = im2.astype(float) / 255.

    s = time.time()
    u, v, im2W = pyflow.coarse2fine_flow(
        im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
        nSORIterations, colType)
    e = time.time()
    print('Time Taken: %.2f seconds for image of size (%d, %d, %d)' % (
        e - s, im1.shape[0], im1.shape[1], im1.shape[2]))
    flow = np.concatenate((u[..., None], v[..., None]), axis=2)
    
    # np.save('examples/outFlow.npy', flow)

    hsv = np.zeros(im1.shape, dtype=np.uint8)
    hsv[:, :, 0] = 255
    hsv[:, :, 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # print('save of to %s' % fname)
    save_img(rgb, fname)
    # cv2.imwrite('examples/car2Warped_new.jpg', im2W[:, :, ::-1] * 255)


def save_img(im, fname): 
    cv2.imwrite(fname, im)


def get_name(p):
    if '/' in p:
        return p.split('/')[-1].split('.')[0]
    return p.split('.')[0]


def mk_folder(vid_src, dst_dir):
    dst_folder = dst_dir + get_name(vid_src)
    if not os.path.exists(dst_folder):
        os.mkdir(dst_folder)
    return dst_folder


def crop_img(img, size):
    left, up, right, bottom = map(int, size.split(','))
    h, w, _ = img.shape
    l = max(0, left)
    u = max(0, up)
    r = min(w - 1, right)
    b = min(h - 1, bottom)

    assert (l < r and b > u)
    return img[u:b, l:r]
    

def extract_of(vid_src, img_dir, of_dir, crop=None):
    img_dir = mk_folder(vid_src, img_dir)
    print('img dir: %s' % img_dir)
    of_dir = mk_folder(vid_src, of_dir)
    print('of dir: %s' % of_dir)

    vidcap = cv2.VideoCapture(vid_src)
    success,image = vidcap.read()
    cnt = 0

    prev_img = None 

    while True:
        success,image = vidcap.read()
        if not success:
            break 

        if cnt % 10 == 0:
            print('----- fid: %d' % cnt)

        if crop:
            image = crop_img(image, crop)
        
        # print(image.shape)
        # break 

        resized_image = cv2.resize(image, DST_SIZE) 

        if cnt > 0:
            fname = '/{:05d}.png'.format(cnt)
            save_of(prev_img, resized_image, of_dir + fname)
            save_img(resized_image, img_dir + fname)

        prev_img = resized_image.copy()
        cnt += 1


extract_of( args.vid_src, 
            args.img_dir,
            args.of_dir,
            args.crop)