import csv
import numpy as np
import cv2

def read_data(file_path,header=True,delimiter=','):
    # The read-in data should be a N*W matrix,
    # where N is the length of the time sequences,
    # W is the number of sensors/data features
    i = 0
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter = delimiter)
        data=[]
        for line in reader:
            if i == 0 and header:
                i += +1
            else:
                line = np.array(line, dtype = 'float') # str2float
                if i == 0  or (i == 1 and header):
                    data = line
                else:
                    data = np.vstack((data, line))
                i += 1
    return data

def read_binary(file_path,header=True,delimiter=','):
    # The read-in data should be a N*W matrix,
    # where N is the length of the time sequences,
    # W is the number of sensors/data features
    i = 0
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter = delimiter)
        data=[]
        for line in reader:
            if i == 0 and header:
                i += +1
            else:
                for j, element in enumerate(line):
                    if element == 'True':
                        line[j] = True
                    elif element == 'False':
                        line[j] = False
                    else:
                        raise ValueError("Data type is not boolean!!")
                    
                line = np.array(line) # str2float
                if i == 0  or (i == 1 and header):
                    data = line
                else:
                    data = np.vstack((data, line))
                i += 1
    return data

def read_human_data(file_path,num_header=2,delimiter='\t'):
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter=delimiter)
        for i,line in enumerate(reader):
            if i >= num_header:
                act = int(line[1])
                x = float(line[3])
                y = float(line[5])
                vx = float(line[7])
    #             vy = float(line[8])
                ax = float(line[9])
    #             ay = float(line[11])
    #             yaw =float([line[13]])
    #             yaw_dot =float([line[14]])
                if i == num_header:
                    data = np.array([act,x,y,vx,ax])
                else:
                    data = np.vstack([data, np.array([act,x,y,vx,ax])])
    return data
