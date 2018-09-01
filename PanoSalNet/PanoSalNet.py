
#TODO: load PanoSalNet and test on one image
#input: prototxt file, caffemodel file, test img file
#output: salient
import cv2
import numpy as np
import caffe
import timeit

def post_filter(_img):
    result = np.copy(_img)
    result[:3,:3] = _img.min()
    result[:3, -3:] = _img.min()
    result[-3:, :3] = _img.min()
    result[-3:, -3:] = _img.min()
    return result

net = caffe.Net('./panosalnet/panosalnet_test.prototxt', caffe.TEST)
net.copy_from('./panosalnet/panosalnet_iter_800.caffemodel')

C, H, W = net.blobs['data1'].data[0].shape

img_filepath = './panosalnet/test.png'
img = cv2.imread(img_filepath)
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
