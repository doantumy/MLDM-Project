import numpy as np
import cv2
import time
import freeman as fm
import collections
import edit_distance as ed
from fuzzywuzzy import fuzz
from sklearn.metrics import pairwise_distances_argmin



def process_image(file_name):
    image = cv2.imread(file_name, 0)
    image_resize = cv2.resize(image, (26,26))
    new_image = cv2.copyMakeBorder(image_resize, 1,1,1,1, cv2.BORDER_CONSTANT, value=[255,255,255])
    invert_image = cv2.bitwise_not(new_image)
    return invert_image


def normalize_data(input_data):
    data = input_data.astype(np.float32)
    return np.multiply(data, 1.0/255.0)

def knn(test_img, X, y, k, kstrip=10):
    tic = time.time()
    distances = []
    for image in X:
        distances.append(ed.edit_distance(test_img, image, kstrip).cal_distance()[0].astype(int)) 
    cls = [y for _,y in sorted(zip(distances, y))[:k]]
    counter = collections.Counter(cls)
    toc = time.time()
    print("Time: ", toc-tic)
    return counter.most_common(1)[0][0], toc-tic

def kmeans_knn(test_img, centroids, cluster_lbl, X_freeman, y_label, kstrip=10, nn=5, show=False, fm_mode='normal'):
    # Find nearest centroid of test image (1,784) array
    nearest_centroid = pairwise_distances_argmin(test_img.reshape(1,-1), centroids)
    print("Nearest centroid: ",nearest_centroid)
    # Convert test image to freeman code
    test_img_reshape = test_img.reshape((28, 28))
    test_freeman, _ = fm.freeman_chain_code(np.float32(test_img_reshape),fm_mode)
    test_fm_code = ''.join(test_freeman)
    # Get all freeman of X set and y labels for all examples in nearest cluster
    X_freeman_array = np.asarray(X_freeman)
    new_freeman_cluster = X_freeman_array[cluster_lbl==nearest_centroid]
    new_y_cluster = y_label[cluster_lbl==nearest_centroid]
    # Calculate KNN for test freeman and examples in cluster freemans
    pred, time = knn(test_fm_code, new_freeman_cluster, new_y_cluster, kstrip, nn)
    return pred, time


def find_subseq_match(freeman_code, freq_sequence, freeman_boundaries):
    i = len(freeman_code)
    j = len(freq_sequence)
    index = -1
    max_sim = 0
    max_sim_fm = ""
    sub_boundaries = []
    if i < j: return 0, 0, freeman_code
    else:
        for k in range(0, i-j):
            sub = freeman_code[k:k+j]
            bound_sub = freeman_boundaries[k:k+j]
            #print(sub)
            sim = fuzz.ratio(freq_sequence, sub)
            if  sim > max_sim: 
                max_sim = sim
                #print(max_sim)
                index = k
                #print(k)
                max_sim_fm = sub
                #print(max_sim_fm)
                sub_boundaries = bound_sub
            #print("sim", sim, "max_sim", max_sim, "index", index)
    return index, max_sim, sub_boundaries, max_sim_fm