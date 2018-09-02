#Write a script to get head position of all paris video
#record position in time around 10 seconds

#see what fixation in 10th seconds
import numpy as np
import os



def parse_raw(_file_name):#for 360 ds log
  temp = open(_file_name).read().split('\n')[:-1]
  temp2 = [map(float, item.split(' ')) for item in temp]
  for i, _ in enumerate(temp2):
    item = temp2[i]
    temp2[i] = [item[0], item[1], item[3], item[5], item[4], item[2]]
  return np.array(temp2)

def parse_raw2(_file_name):#for 360 ds log
    temp = open(_file_name).read().split('\n')[1:-1]#remove header and useless last line
    temp2 = [map(float, item.split(',')[1:]) for item in temp]
    #timestamp_list = [datetime.datetime.strptime(item.split(',')[0], "%Y-%m-%d %H:%M:%S.%f") for item in temp]
    for i, _ in enumerate(temp2):
        item = temp2[i]#timestamp, z, y, x, w, ....
        temp2[i] = [item[0], -1, item[3], item[2], item[1], item[4]]
    return np.array(temp2)

def get_series_ds(_file_path, _topic):
    file_name_list = []
    series_ds = []
    for root, dirs, files in os.walk(_file_path):
        for file in files:
            if file.endswith(".txt") and file.lower().find(_topic) >= 0:
                 file_name_list.append((os.path.join(root, file)))

    for idx, file_name in enumerate(file_name_list):
        series = parse_raw(file_name);
        series_ds.append(series.tolist())
    series_ds = np.array(series_ds)
    return series_ds

def get_series_ds2(_file_path, _topic, _dataset=0):
    file_name_list = []
    series_ds = []
    f_parse=None
    file_sig=''
    if _dataset==0:
        f_parse=parse_raw
        file_sig='.txt'
    elif _dataset==1:
        f_parse=parse_raw2
        file_sig='.csv'
    
    for root, dirs, files in os.walk(_file_path):
        for file in files:
            if file.endswith(file_sig) and file.lower().find(_topic) >= 0:
                 file_name_list.append((os.path.join(root, file)))

    for idx, file_name in enumerate(file_name_list):
        series = f_parse(file_name);
        series_ds.append(series.tolist())
    series_ds = np.array(series_ds)
    return series_ds

def retrieve(_time = 10.0, _topic='paris'):
    series_ds = get_series_ds(u'/Users/phananh/Desktop/GSU/Datasets/Dataset1/results/', _topic)
    #get ALL position at time + dt, 
    dt = 1.0/30
    series_dt = []
    for series in series_ds:
      temp = []
      for item in series: 
        if item[0] >= _time and item[0] <= _time + dt:
          temp.append(item)
      series_dt.append(temp)
    #get quaternion from the first elements of each users
    q_list = [item[0][2:] for item in series_dt if item != []]
    return q_list

def extract_direction(_q, v0=[1, 0, 0]):
    return _q.rotate(v0)

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))/np.pi * 180
    #return math.acos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))/np.pi * 180
    
def get_speed(v_list, timestamp_list, idx, d=7):
    #TODO: calculate speed of head orientation
    #INPUT: list of head orientation and associated timestamp
    #OUPUT: speed of current head orientation
    theta = angle_between(v_list[idx], v_list[max(idx-d, 0)])
    return

def cutoff_vel_acc(_vector_ds):
    dd = 7
    stats_ds = []
    for vec in _vector_ds:
        stats = []
        for idx in range(len(vec)):
            if idx < 5:
                continue
            dt = vec[idx][0] - vec[idx-dd][0]
            theta = angle_between(vec[idx][1], vec[idx-dd][1])
            v = theta * 1.0 / dt   
            vec[idx][2] = v
            
            #if idx == 1:
            #   continue
            dv = vec[idx][2] - vec[idx-dd][2]
            a = dv * 1.0 / dt
            vec[idx][3] = a
            item = [vec[idx][0], vec[idx][1], v, a]
            stats.append(item)
        stats_ds.append(stats)
    result = []
    for vector in stats_ds:
        #removing too fast movement
        remove_idx = set()
        collect_mode = 0 #0 is normal, 1 is begin fast, 2 is begin slow
        #print stats_ds[0]
        for idx, (timestamp, vec, v, a) in enumerate(vector):
            if v > 20 and a > 50:
                collect_mode = 1;
            if a < -40 and collect_mode == 1:#slowing down
                collect_mode = 2
            if collect_mode == 2 and a > -50  and v < 20:#slowing down finish
                collect_mode = 0
            if collect_mode == 1 or collect_mode == 2:
                remove_idx.add(idx)
        result.append([vector[idx] for idx,_ in enumerate(vector) if idx not in remove_idx])
    return result
    
def get_vector_ds(_topic, _dataset=0, cutoff=True):
    from pyquaternion import Quaternion
    
    if _dataset==0:
        file_path = u'/home/u9168/salnet/Datasets/Dataset1/results/'
    elif _dataset==1:
        file_path = u'/home/u9168/salnet/Datasets/Dataset2/Formated_Data/Experiment_1/'
        
    series_ds = get_series_ds2(file_path, _topic, _dataset=_dataset)
    vector_ds = []
    for series in series_ds:
        vec = []
        #for item in series:
        for idx in np.arange(0, len(series)):
            item = series[idx]
            if _dataset==0:
                v = extract_direction(Quaternion([item[5], item[4], item[3], item[2]]))#
            elif _dataset==1:
                v = extract_direction(Quaternion([item[5], -item[4], item[3], -item[2]]), v0=[0, 0, 1])#
            vec.append([item[0], v, 0, 0])#time, cur pos, angular vec, angular acc
        vector_ds.append(vec)
    
    if cutoff==True:
        result = cutoff_vel_acc(vector_ds)
    else:
        result = vector_ds
    return result
    #success! the code now show some acceleration and speed!
    #let remove all position excess threshold, select some random frame
    #to create DATASET!!!!!

def get_fixation(_vector_ds, _time):
    dt = 1.0/30
    series_dt = []
    for vector in _vector_ds:
        temp = []
        for item in vector: 
            if item[0] >= _time - 3*dt and item[0] <= _time + 1*dt:
                temp.append(item)
        series_dt.append(temp)
    #get quaternion from the first elements of each users
    result = []
    for series in series_dt:
        for item in series:
            result.append(item)
    #result = [item[0] for item in series_dt if item != []]
    return result

#example: 
#get a dataset of vectors topic paris
#then get fixation at given time t = 5 seconds
#
#vector_ds = get_vector_ds(_topic='paris')
#get_fixation(vector_ds, 5)
#
##succesfully eliminate saccade
#now calculate human perfomance 
#get 100 fixation, 58 users each
#get salient for one user, calculate %cover & TPR, at 100 fixation
# step 1: generate salient map
# step 2: compare with n-1 users fixation
# report results







