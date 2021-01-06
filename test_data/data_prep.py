import cv2
import os
import numpy as np

rawdir = 'target'
rawdata = os.listdir(rawdir)

outdir = 'input'

# PRE PREP
for i in range(len(rawdata)):
    # open
    img = cv2.imread(os.path.join(rawdir, rawdata[i]))
    # resize
    img = cv2.resize(img, (256,256))
    # overwrite
    cv2.imwrite(os.path.join(rawdir, rawdata[i]), img)
    # make canny
    edges = cv2.Canny(img,210,240)
    # save canny
    cv2.imwrite(os.path.join(outdir, rawdata[i]), edges)


