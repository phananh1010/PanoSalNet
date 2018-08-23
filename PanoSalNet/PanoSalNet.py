
#TODO: load PanoSalNet and test on one image
import caffe
import pickle
import cv2
import numpy as np
from matplotlib import pyplot as plt


def post_filter(_img):
    result = np.copy(_img)
    result[:3,:3] = _img.min()
    result[:3, -3:] = _img.min()
    result[-3:, :3] = _img.min()
    result[-3:, -3:] = _img.min()
    return result

if 'solver' not in locals():
    solver = caffe.SGDSolver('./PanoSalNet_solver.prototxt')
    #solver.net.copy_from('./states/salnetv6_step_iter_800.caffemodel')
    solver.test_nets[0].copy_from('./states/salnetv6_step_iter_800.caffemodel')
    
mu = [ 104.00698793, 116.66876762, 122.67891434]
input_scale = 0.0078431372549    
img0 = cv2.imread(img_file_path)
img0 = img0.astype(np.float32)
img0 -= np.array(mu)
img0 *= input_scale

img = np.transpose(cv2.resize(np.transpose(img0, (1, 2, 0)), (384, 216)), (2, 0, 1))
img = img.reshape(1, *img.shape)

solver.test_nets[0].blobs['data1'].data[0] = img

solver.test_nets[0].forward(start='conv1')
fx = solver.test_nets[0].blobs['deconv1'].data[0][0]
fx = post_filter(fx)
