import os
import cv2
import argparse
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(
            prog='NumPy PNG Joiner',
            description='Join numpy images into side-by-side PNGs'
        )
parser.add_argument('input',default='out',
                    help='Directory to parse. Default `out\'.')
parser.add_argument('output',default='png_out',
                    help='Directory to place PNGs into. Default `png_out\'')
args = parser.parse_args()

t_res = 256
subsets = ['train','val','test']
subtype = ['CT','MR']

def iload(path):
    img = np.moveaxis(np.load(path),0,-1)
    img = cv2.resize(img,(t_res,t_res))
    return img

for c_sset in subsets:
    i0_dir = os.path.join(args.input, c_sset, subtype[0])
    i1_dir = os.path.join(args.input, c_sset, subtype[1])
    o_dir = os.path.join(args.output, c_sset)
    if not os.path.exists(o_dir):
        os.makedirs(o_dir)
    img = np.zeros((t_res,2*t_res))
    for c_file in os.listdir(i0_dir):
        img *= 0
        im0 = iload(os.path.join(i0_dir,c_file))
        im1 = iload(os.path.join(i1_dir,c_file))
        img[:,:t_res] = im0
        img[:,t_res:] = im1
        pimg = Image.fromarray((255*img).astype(np.uint8))
        pimg.save(os.path.join(o_dir, c_file[:-4] + '.png'), "PNG")
