#TODO: predict head movement in next 0.5 second
#input: list of salient maps from file named sds
#       list of head position from file name series
#output: list of prediction head position map

import keras
import numpy as np
import pickle
from keras import models
from sklearn import preprocessing
import get_fixation
import cv2
import re


def vector_to_ang(_v):
    #v = np.array(vector_ds[0][600][1])
    #v = np.array([0, 0, 1])
    _v = np.array(_v)
    alpha = get_fixation.degree_distance(_v, [0, 1, 0])#degree between v and [0, 1, 0]
    phi = 90.0 - alpha
    proj1 = [0, np.cos(alpha/180.0 * np.pi), 0] #proj1 is the projection of v onto [0, 1, 0] axis
    proj2 = _v - proj1#proj2 is the projection of v onto the plane([1, 0, 0], [0, 0, 1])
    theta = get_fixation.degree_distance(proj2, [1, 0, 0])#theta = degree between project vector to plane and [1, 0, 0]
    sign = -1.0 if get_fixation.degree_distance(_v, [0, 0, -1]) > 90 else 1.0
    theta = sign * theta
    return theta, phi

def ang_to_geoxy(_theta, _phi, _h, _w):
    x = _h/2.0 - (_h/2.0) * np.sin(_phi/180.0 * np.pi)
    temp = _theta
    if temp < 0: temp = 180 + temp + 180
    temp = 360 - temp
    y = (temp * 1.0/360 * _w)
    return int(x), int(y)
    

def create_fixation_map(_X, _y, _idx):
    v = _y[_idx]
    theta, phi  = vector_to_ang(v)
    hi, wi = ang_to_geoxy(theta, phi, H, W)
    result = np.zeros(shape=(H, W))
    result[H-hi-1, W-wi-1] = 1
    return result

def create_dataset(dataset, look_back=1, look_ahead=1):
    #dataset has shape NxC
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-look_ahead-1):
        a = dataset[i:(i+look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back + look_ahead, :])
    return np.array(dataX), np.array(dataY)

sds = pickle.load(open("./GitHub_misc/sds"))
series = pickle.load(open("./GitHub_misc/series"))

N, H, W = sds.shape
gblur_size= 5
look_back = 15
look_ahead = 8
mmscaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
modelname = '360net'
model3 = models.load_model('./GitHub_misc/model3_{}_128_w16_h9_4000'.format(modelname))
print model3.summary()

headmap = np.array([create_fixation_map(None, series, idx) for idx,_ in enumerate(series)])
headmap = np.array([cv2.GaussianBlur(item, (gblur_size, gblur_size), 0) for item in headmap])
headmap = mmscaler.fit_transform(headmap.ravel().reshape(-1, 1)).reshape(headmap.shape)

ds = np.zeros(shape=(N, 2, H, W))
ds[:, 0, :, :] = sds
ds[:, 1, :, :] = headmap
Xtest, ytest = create_dataset(ds, look_back=look_back, look_ahead=look_ahead)
ytest = series[-len(Xtest):] 

fx = model3.predict(Xtest.reshape(-1, look_back, 2*H*W)).reshape(-1, H, W)
