#TODO: extract frame by frame movie, then create the 

topic_dict = {'roller': 'roller65.webm', 'paris': 'paris.mp4', '1': 'skiing.mp4', '0': 'conan1.mp4', '3': 'conan2.mp4', '2': 'alien.mp4', '5': 'war.mp4', '4': 'surfing.mp4', '7': 'football.mp4', '6': 'cooking.mp4', '8': 'rhinos.mp4'}
movie_template = '/home/u9168/salnet/{}'
sl_template = '/home/u9168/caffe_model/data/salient_map/{}_done'

import cv2
import lmdb
import numpy as np
import caffe


def extract_frame_at_time(filename, timestamp, width=512, height=288):
    vcap = cv2.VideoCapture(filename)#roller65.webm, paris.mp4; ocean40.webm; venise.webm
    vcap.set(cv2.cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    #width = vcap.get(cv2.cv2.CAP_PROP_FRAME_WIDTH)   # float
    #height = vcap.get(cv2.cv2.CAP_PROP_FRAME_HEIGHT) # float
    #width = int(width * _ratio)
    #height = int(height * _ratio)

    res, frame = vcap.read()
    frameG = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frameS = cv2.resize(frameG, (width, height))
    return frameS

mu = [ 104.00698793, 116.66876762, 122.67891434]
input_scale = 0.0078431372549
h2, w2 = 216, 384#288, 512#36, 64#

#im_train_db = lmdb.open('./image_train_lmdb', map_size=int(2e10))
im_test_db = lmdb.open('./image_test_lmdb2', map_size=int(2e10))
with im_test_db.begin(write=True) as im_test_txn:
    for idx, timestamp, topic in salient_test_list:
        if idx % 1000 == 0: print idx

        file_path = movie_template.format(topic_dict[topic])
        image = extract_frame_at_time(file_path, timestamp, w2, h2)
        cv2.imwrite('./image_test/{}.png'.format(idx), image)
        image = image.astype(np.float32)
        image -= np.array(mu)#mean values
        image *= input_scale
        image = image.transpose((2,0,1))

        im_dat = caffe.io.array_to_datum(image)
        im_test_txn.put('{:0>10d}'.format(idx), im_dat.SerializeToString())
im_test_db.close()
