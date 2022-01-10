from sklearn.mixture import GaussianMixture
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

start_time = timeit.default_timer()  # 시작 시간 체크

# cluster the pixel intensities
k = 5
# n_components로 미리 군집 개수 설정
gmm = GaussianMixture(n_components=k, random_state=42)
gmm_labels = gmm.fit_predict(image)

terminate_time = timeit.default_timer()  # 종료 시간 체크
print("%f초 걸렸습니다." % (terminate_time - start_time))

print('Cluster Label 유형:',np.unique(gmm_labels))

RGB = pd.DataFrame(data=image, columns=feature_names)
labels = gmm_labels
RGB['kmeans_cluster'] = labels
print(RGB)

# 각 레이블의 백분율 구하기
labels_count = RGB['kmeans_cluster'].value_counts()
# print(labels_count)
labels_values = labels_count.values.tolist()
# print(labels_values)

# 배경 제외 4가지 색상 pie chart
hist = v.centroid_histogram(labels)
center_colors = gmm.means_
print(center_colors)
cmyk_list = v.cmyk_op(center_colors)
# print(cmyk_list)

dic = []
for i in range(k):
    dic.append([hist[i], center_colors[i], cmyk_list[i]])
dic.sort(reverse=True)
# print(dic)

cmyk_colors = []
hex_colors = []
rgb_colors = []
i = 0
for persent, color, cmyk_c in dic:
    if round(color[0]) == 255 and round(color[1]) == 255 and round(color[2]) == 255:
        print("background : {:.0f}%".format(persent*100))
        labels_values.pop(i)
        continue
    i+=1
    rgb_colors.append(v.RGB(color))
    hex_colors.append(v.RGB2HEX(color))
    cmyk_colors.append(v.CMYK(cmyk_c))

percentage = labels_values

v.pie_chart(img, cmyk_colors, hex_colors, rgb_colors, percentage)

# Segmentation
v.segmented_image(img, center_colors, labels)

