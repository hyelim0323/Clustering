# import the necessary packages
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import os
import sys
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import time

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)
from common import visualize as v

# load the image and convert it from BGR to RGB so that
# we can dispaly it with matplotlib
img = cv2.imread(os.path.join(ROOT_DIR, "dataset/check_shirts.jpg")) ### 원하는 이미지 경로 설정
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# # show our image
# plt.figure()
# plt.axis("off")
# plt.imshow(img)

# reshape the image to be a list of pixels
image = img.reshape(img.shape[0] * img.shape[1], 3)

# 보다 편리한 데이타 Handling을 위해 DataFrame으로 변환
feature_names = ['R', 'G', 'B']

# cluster the pixel intensities
start = time.time()
k = 5
clt = KMeans(init='k-means++', n_clusters = k)
clt.fit(image)
print(time.time()-start)
RGB = pd.DataFrame(data=image, columns=feature_names)
labels = clt.labels_
RGB['kmeans_cluster'] = labels
print(RGB)

# 각 레이블의 백분율 구하기
labels_count = RGB['kmeans_cluster'].value_counts()
# print(labels_count)
labels_values = labels_count.values.tolist()
# print(labels_values)

# 배경 제외 4가지 색상 pie chart
hist = v.centroid_histogram(labels)
center_colors = clt.cluster_centers_
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
    if color[0] > 254 and color[1] > 254 and color[2] > 254:
        print("background : {:.0f}%".format(persent*100))
        labels_values.pop(i)
        print(labels_values)
        continue
    i+=1
    rgb_colors.append(v.RGB(color))
    hex_colors.append(v.RGB2HEX(color))
    cmyk_colors.append(v.CMYK(cmyk_c))

percentage = labels_values

v.pie_chart(img, cmyk_colors, hex_colors, rgb_colors, percentage)

# Segmentation
v.segmented_image(img, center_colors, labels)

# # pairplot with Seaborn
# sns.pairplot(RGB,hue='kmeans_cluster')
# plt.show()

# scatter plot
r = RGB
fig = plt.figure(figsize=(10,10))
ax = Axes3D(fig)
ax.scatter(r['R'],r['G'],r['B'],c=r['kmeans_cluster'],alpha=0.5)
ax.set_xlabel('R')
ax.set_ylabel('G')
ax.set_zlabel('B')
plt.show()
