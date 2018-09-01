#TODO: load the panosalnet model, create salient map from input image
#input: model config file: panosalnet_test.prototxt
#       model weight: 'panosalnet_iter_800.caffemodel
#       input image: test.png

import cv2
import numpy as np
import caffe
import timeit

FILE_MODEL_CONFIG = 'panosalnet_test.prototxt'
FILE_MODEL_WEIGHT = 'panosalnet_iter_800.caffemodel'
FILE_IMAGE = 'test.png'

def post_filter(_img):
    result = np.copy(_img)
    result[:3,:3] = _img.min()
    result[:3, -3:] = _img.min()
    result[-3:, :3] = _img.min()
    result[-3:, -3:] = _img.min()
    return result

net = caffe.Net(FILE_MODEL_CONFIG, caffe.TEST)
net.copy_from(FILE_MODEL_WEIGHT )

C, H, W = net.blobs['data1'].data[0].shape


img = cv2.imread(FILE_IMAGE)
mu = [ 104.00698793, 116.66876762, 122.67891434]
input_scale = 0.0078431372549

dat = cv2.resize(img, (W, H))
dat = dat.astype(np.float32)
dat -= np.array(mu)#mean values
dat *= input_scale
dat = dat.transpose((2,0,1))

net.blobs['data1'].data[0] = dat
net.forward(start='conv1')

salient = net.blobs['deconv1'].data[0][0]
salient = (salient * 1.0 - salient.min())
salient = (salient / salient.max()) * 255
salient = post_filter(salient)
