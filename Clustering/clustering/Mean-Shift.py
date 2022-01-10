# import the necessary packages
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import os
import sys
import timeit

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)
from common import visualize as v

# load the image and convert it from BGR to RGB so that
# we can dispaly it with matplotlib
img = cv2.imread(os.path.join(ROOT_DIR, "dataset/check_shirts.jpg")) ### 원하는 이미지 경로 설정
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# reshape the image to be a list of pixels
image = img.reshape(img.shape[0] * img.shape[1], 3)

# 보다 편리한 데이타 Handling을 위해 DataFrame으로 변환
feature_names = ['R', 'G', 'B']

# # quantile : 데이터 개수의 일정 비율만큼 샘플링하면서 meanshift하게됨
# # 따라서, 데이터 개수가 엄청 많아질시 quantile값이 적으면 시간이 너무 오래걸림
# bandwidth = estimate_bandwidth(image, quantile=0.8)
# print("최적의 bandwidth 값:", bandwidth)
# cluster the pixel intensities
meanshift = MeanShift(bandwidth=5)
# clustering 레이블 반환
cluster_labels = meanshift.fit_predict(image)
print('Cluster Label 유형:',np.unique(cluster_labels))
# print(meanshift.cluster_centers_)

RGB = pd.DataFrame(data=image, columns=feature_names)
labels = meanshift.labels_
RGB['kmeans_cluster'] = labels
print(RGB)

# 각 레이블의 백분율 구하기
labels_count = RGB['kmeans_cluster'].value_counts()
# print(labels_count)
labels_values = labels_count.values.tolist()
# print(labels_values)

# 배경 제외 4가지 색상 pie chart
hist = v.centroid_histogram(labels)
# print(len(hist))
center_colors0 = meanshift.cluster_centers_
# print(len(center_colors))
cmyk_list = v.cmyk_op(center_colors0)
# print(len(cmyk_list))

hist = hist.tolist()
center_colors = center_colors0.tolist()

dic = []
for i in range(5):
    idx = hist.index(max(hist))
    dic.append([hist[idx], center_colors[idx], cmyk_list[idx]])
    hist.pop(idx)
    center_colors.pop(idx)
    cmyk_list.pop(idx)
print(dic)
dic.sort(reverse=True)

cmyk_colors = []
hex_colors = []
rgb_colors = []
i = 0
for persent, color, cmyk_c in dic:
    if color[0] > 254 and color[1] > 254 and color[2] > 254:
        print("background : {:.0f}%".format(persent*100))
        labels_values.pop(i)
        continue
    i+=1
    rgb_colors.append(v.RGB(color))
    hex_colors.append(v.RGB2HEX(color))
    cmyk_colors.append(v.CMYK(cmyk_c))

percentage = labels_values[:4]

v.pie_chart(img, cmyk_colors, hex_colors, rgb_colors, percentage)

# Segmentation
v.segmented_image(img, center_colors0.tolist(), labels)



