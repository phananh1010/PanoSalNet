
import cv2
import numpy as np
import cv2
from Quaternion import Quat, normalize
from pyquaternion import Quaternion
import timeit
from scipy import stats

import get_qlist

from matplotlib import pyplot


#seek_time = 40
source = 'france';#'own', 'france'

topic_dict = {'paris':'paris.mp4', 'diving':'ocean40.webm', 'venise':'venise.webm', 'roller':'roller65.webm', 'timelapse': 'newyork.webm', '0': 'conan1.mp4', '1':'skiing.mp4', '2':'alien.mp4', '3':'conan2.mp4', '4':'surfing.mp4', '5': 'war.mp4','7':'football.mp4', '8': 'rhinos.mp4', '6': 'cooking.mp4'}

topic_list = [['paris', 'diving', 'venise', 'roller', 'timelapse', 'newyork'],\
              ['0', '1', '2', '3', '4', '5', '6', '7', '8']]

topic_info_dict = {'paris': ['paris.mp4', 244.06047, 3840, 2048], 'timelapse': ['newyork.webm', 91.03333, 3840, 2048], '3': ['conan2.mp4', 172.5724, 2560, 1440], '1': ['skiing.mp4', 201.13426, 2560, 1440], '0': ['conan1.mp4', 164.1973, 2560, 1440], 'venise': ['venise.webm', 175.04, 3840, 2160], '2': ['alien.mp4', 293.2333, 2560, 1440], '5': ['war.mp4', 655.0544, 2160, 1080], '4': ['surfing.mp4', 205.7055, 2560, 1440], '7': ['football.mp4', 164.8, 2560, 1440], '6': ['cooking.mp4', 451.12, 2560, 1440], 'diving': ['ocean40.webm', 372.23853, 3840, 2048], 'roller': ['roller65.webm', 69.0, 3840, 2048], '8': ['rhinos.mp4', 292.0584, 2560, 1440]}

#goal: given a fixed time, and vector dataset, prepare the vec_map
#step 1: read the movie, get weight, height
#step 2: from w, h, fixation with timestam, generate salient map


def geoy_to_phi(_geoy, _height):
  d = (_height/2 - _geoy) * 1.0 / (_height/2)
  s = -1 if d < 0 else 1
  return s * np.arcsin(np.abs(d)) / np.pi * 180

def pixel_to_ang(_x, _y, _geo_h, _geo_w):
  phi = geoy_to_phi(_x, _geo_h)
  theta = -(_y * 1.0 / _geo_w) * 360
  if theta < -180: theta = 360 + theta
  return theta, phi

def extract_direction(_q):
  v = [1, 0, 0]
  return _q.rotate(v)

#CALCULATE DEGREE DISTANCE BETWEEN TWO 3D VECTORS
def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def degree_distance(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))/np.pi * 180
#def degree_distance(_q0, _q1):
#  #input is directional vector without roll v = [v0, v1, v2]
#  return np.arccos(np.dot(_q0, _q1)/np.linalg.norm(_q0)/np.linalg.norm(_q1)) / np.pi * 180

def gaussian_from_distance(_d, _gaussian_dict):
  temp = np.around(_d, 1)
  return _gaussian_dict[temp] if temp in _gaussian_dict else 0.0

def create_pixel_vecmap(_geo_h, _geo_w):
  vec_map = np.zeros((_geo_h, _geo_w)).tolist()
  for i in range(_geo_h):
    for j in range(_geo_w):
      theta, phi = pixel_to_ang(i, j, _geo_h, _geo_w)
      t = Quat([0.0, theta, phi]).q #nolonger use Quat
      q = Quaternion([t[3], t[2], -t[1], t[0]])
      vec_map[i][j] = extract_direction(q)
  return vec_map



def init(_topic, _seek_time, _var, _ratio=1.0/10):
    gaussian_dict = {np.around(_d, 1):stats.multivariate_normal.pdf(_d, mean=0, cov=_var) for _d in np.arange(0.0, 180, .1  )}
    
    video_name = topic_dict[_topic]
    vcap = cv2.VideoCapture(video_name)#roller65.webm, paris.mp4; ocean40.webm; venise.webm
    vcap.set(cv2.cv2.CAP_PROP_POS_MSEC, _seek_time * 1000)
    width = vcap.get(cv2.cv2.CAP_PROP_FRAME_WIDTH)   # float
    height = vcap.get(cv2.cv2.CAP_PROP_FRAME_HEIGHT) # float
    width = int(width * _ratio)
    height = int(height * _ratio)

    res, frame = vcap.read()
    frameG = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameS = cv2.resize(frameG, (width, height))
    
    vec_map = create_pixel_vecmap(height, width)

    return width, height, frameS, vec_map, gaussian_dict

def create_salient(_fixation_list, _vec_map, _width, _height, _gaussian_dict, verbal=False):
    idx = 0
    heat_map = np.zeros((_height, _width))
    for i in range(heat_map.shape[0]):
        for j in range(heat_map.shape[1]):
            qxy = _vec_map[i][j]
            for fixation in _fixation_list:
                q0 = fixation[1] 
                btime = timeit.default_timer()
                d = degree_distance(q0, qxy)

                dd_time = timeit.default_timer() - btime

                heat_map[i, j] += 1.0 * gaussian_from_distance(d, _gaussian_dict)
                gau_time = timeit.default_timer() - btime - dd_time

            idx += 1;
            if verbal == False: continue
            if idx % 10000 == 0:
                  print _width * _height, idx, i, j, heat_map[i, j], d, dd_time, gau_time
            if d < 5: 
                  print '<5 degree: ---->', _width * _height, idx, i, j, heat_map[i, j], d, dd_time, gau_time
    return heat_map

def create_salient_list(_fixation_list, _vec_map, _width, _height, _gaussian_dict, verbal=False):
    #same as create_salient, but with indivial fixation
    #sum all together heat_map will result in heat_map in create_salient
    idx = 0
    heat_map_list = np.zeros((len(_fixation_list), _height, _width))
    for i in range(heat_map_list[0].shape[0]):
        for j in range(heat_map_list[0].shape[1]):
            qxy = _vec_map[i][j]
            for k, fixation in enumerate(_fixation_list):
                q0 = fixation[1] 
                btime = timeit.default_timer()
                d = degree_distance(q0, qxy)

                dd_time = timeit.default_timer() - btime

                heat_map_list[k, i, j] += 1.0 * gaussian_from_distance(d, _gaussian_dict)
                gau_time = timeit.default_timer() - btime - dd_time

            idx += 1;
            if verbal == False: continue
            if idx % 10000 == 0:
                  print _width * _height, idx, i, j, heat_map[i, j], d, dd_time, gau_time
            if d < 5: 
                  print '<5 degree: ---->', _width * _height, idx, i, j, heat_map[i, j], d, dd_time, gau_time
    return heat_map_list

def get_frame_at_time(_topic, _seek_time, _ratio=.1, isgray=True):
    video_name = topic_dict[_topic]
    vcap = cv2.VideoCapture(video_name)#roller65.webm, paris.mp4; ocean40.webm; venise.webm
    vcap.set(cv2.cv2.CAP_PROP_POS_MSEC, _seek_time * 1000)
    width = vcap.get(cv2.cv2.CAP_PROP_FRAME_WIDTH)   # float
    height = vcap.get(cv2.cv2.CAP_PROP_FRAME_HEIGHT) # float
    width = int(width * _ratio)
    height = int(height * _ratio)

    res, frame = vcap.read()
    if isgray == True:
        frameG = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        frameG = frame
    frameS = cv2.resize(frameG, (width, height))
    
    return frameS 


#example usage
#seek_time = 5.1
#var = 20
#width, height, frameS, vec_map, gaussian_dict = init('paris', seek_time, var, _ratio=.1)
#vector_ds = get_qlist.get_vector_ds(_topic='paris')
#fixation_list = get_qlist.get_fixation(vector_ds, seek_time)
#heat_map = create_salient(fixation_list, vec_map, width, height, gaussian_dict)
#pyplot.imshow(heat_map)
#pyplot.show()


#get 20% fixation, create salient map
#




#
