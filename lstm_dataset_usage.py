
#This python file explains steps taken to read the user head orientation vectors and saliency maps from file "salient_ds_dict_w16_h9"
#NOTE: this code assumes the file "salient_ds_dict_w16_h9" is in current working directory

import pickle
import numpy as np

#load data from pickle file. 
if 'salient_ds_dict' not in locals():
    with open('salient_ds_dict_w16_h9', 'rb') as file_in:
        salient_ds_dict = pickle.load(file_in)

#head movement and saliency maps are stored by video names in salient_ds_dict['360net']
#to show the available video names, print all the keys:
print 'list of video names in dataset: ', salient_ds_dict['360net'].keys()

#there are four types of data for each video: 
# timestamp: associated timestamp
# salienct: list of saliency maps size (9x16)
# saliency_filename: list of video frames with video time
# headpos: list of head orientation vectors

#example of getting head orientation and saliency maps of video name='5', user id=14, at video time 152.189
VID_NAME = '5'
UID = 14
TIME_INDEX = 3 #index=3 is associated with video time 152.189

saliency_map = salient_ds_dict['360net'][VID_NAME]['salient'][UID][TIME_INDEX]
headpos = salient_ds_dict['360net'][VID_NAME]['headpos'][UID][TIME_INDEX]

print 'saliency map & head position associated with video "5" at time t=152.189'
print 'head position: {}'.format(headpos)
print 'saliency_map: {}'.format(saliency_map)