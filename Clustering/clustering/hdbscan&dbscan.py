import hdbscan
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

RGB = pd.DataFrame(data=image, columns=feature_names)

# hdbscan
start_time = timeit.default_timer()  # 시작 시간 체크

epsilon = 0.5
clusterer = hdbscan.HDBSCAN(min_cluster_size=60, min_samples=5, cluster_selection_epsilon=epsilon)
dbscan_labels = clusterer.fit_predict(image)

terminate_time = timeit.default_timer()  # 종료 시간 체크
print("%f초 걸렸습니다." % (terminate_time - start_time))

print('Cluster Label 유형:',np.unique(dbscan_labels))

RGB['dbscan_cluster'] = dbscan_labels
print(RGB)

# 각 레이블의 백분율 구하기
labels_count = RGB['dbscan_cluster'].value_counts()
# print(labels_count)
labels_list = labels_count.index.tolist()
# print(labels_list)
labels_values = labels_count.values.tolist()
labels_total = labels_count.values.sum()
# print(labels_total)
labels_per = []
for i in range(len(labels_values)):
    labels_per.append(labels_values[i] / labels_total)
# print(labels_per)

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise_ = list(dbscan_labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

# 각 군집의 평균 RGB 값 구하기
r_mean = RGB.groupby('dbscan_cluster')['R'].mean()
#print(r_mean)
g_mean = RGB.groupby('dbscan_cluster')['G'].mean()
# print(g_mean)
b_mean = RGB.groupby('dbscan_cluster')['B'].mean()
# print(b_mean)
rgb_result = pd.concat([r_mean, g_mean, b_mean], axis=1)
# print(rgb_result)
# print(type(rgb_result))

# 색깔과 퍼센트
# print("Color & Percentage")
# for i in range(n_clusters_):
#     index = labels_list[i]
#     print("RGB : {:.0f}, {:.0f}, {:.0f} / {:.0f}%".format(r_mean[index], g_mean[index], b_mean[index], labels_per[i]*100))

# 배경 제외 상위 4가지 색상 pie chart
best_4 = []
i = 0
print("Percentage List / ")
print(labels_per)
while len(best_4) < 4:
    index = labels_list[i]
    if round(r_mean[index]) == 255 and round(g_mean[index]) == 255 and round(b_mean[index]) == 255:
        print("background : {:.0f}%".format(labels_per[i] * 100))
        labels_values.pop(i)
        continue
    # print(r_mean[index], g_mean[index], b_mean[index])
    best_4.append([r_mean[index], g_mean[index], b_mean[index]])
    i += 1

best_4 = np.array(best_4, dtype=np.float64)
cmyk= v.cmyk_op(best_4)

percentage = labels_values[:4]
cmyk_colors = []
hex_colors = []
rgb_colors = []
for color, cmyk_c in zip(best_4, cmyk):
    rgb_colors.append(v.RGB(color))
    hex_colors.append(v.RGB2HEX(color))
    cmyk_colors.append(v.CMYK(cmyk_c))

v.pie_chart(img, cmyk_colors, hex_colors, rgb_colors, percentage)

# Segmentation
centers = rgb_result.values.tolist()
first = centers.pop(0)
centers.append(first)
v.segmented_image(img, centers, dbscan_labels)



