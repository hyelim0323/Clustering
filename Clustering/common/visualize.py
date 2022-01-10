import matplotlib.pyplot as plt
import numpy as np
import cv2

# 각 label의 백분율
def centroid_histogram(labels):
	# grab the number of different clusters and create a histogram
	# based on the number of pixels assigned to each cluster
	numLabels = np.arange(0, len(np.unique(labels)) + 1)
	(hist, _) = np.histogram(labels, bins = numLabels)
	# normalize the histogram, such that it sums to one
	hist = hist.astype("float")
	hist /= hist.sum()
	# return the histogram
	return hist

# bar 색상표
def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0
    # loop over the percentage of each cluster and the color of
    # each cluster
    # dic = []
    # for i in range(len(hist)):
    #     dic.append([hist[i], centroids[i]])
    # dic.sort(reverse=True)
    # print(dic)

    # for percent, color in dic:
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar

# CMYK 값 구히기
def cmyk_op(rgb):
    result_cmyk = []
    for one_rgb in rgb:
        rgb1= one_rgb / 255.0
        k = 1 - max(rgb1)
        c = (1 - rgb1[0] - k) / (1 - k) * 100
        m = (1 - rgb1[1] - k) / (1 - k) * 100
        y = (1 - rgb1[2] - k) / (1 - k) * 100
        result_cmyk.append([c, m, y, k * 100])
    return result_cmyk

# CMYK
def CMYK(color):
    return "{},{},{},{}".format(int(color[0]), int(color[1]), int(color[2]), int(color[3]))

# HEX
def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

# RGB
def RGB(color):
    return "{},{},{}".format(int(color[0]), int(color[1]), int(color[2]))

# pie_chart
def pie_chart(img, cmyk_colors, hex_colors, rgb_colors, percentage):
    count = str(len(percentage))
    # plt.rcParams["figure.figsize"] = (20, 10)
    fig = plt.figure()

    # ax1 = fig.add_subplot(1, 4, 1)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title("(5) " + count + " Colors Detection", fontsize=30)
    plt.axis("off")
    plt.imshow(img)

    # ax2 = fig.add_subplot(1, 4, 2)
    ax2 = fig.add_subplot(1, 2, 2)
    plt.title('CMYK', fontsize=20)
    # plt.pie(percentage, labels=cmyk_colors, colors=hex_colors, autopct="%.0f%%")
    _, _, autotexts = plt.pie(percentage, labels = cmyk_colors, colors = hex_colors, autopct="%.0f%%", textprops={'fontsize': 20})
    print(count + " Colors Detection/")
    percent = []
    for autotext in autotexts:
        percent.append(autotext.get_text().rstrip("%"))
        # print(autotext.get_text())
        autotext.set_color("red") ### 색상 지정

    print("Main Color : CMYK(" + cmyk_colors[0] + ")")
    print("Sub Color :", end=" ")
    for i in range(1, len(percent)):
        # if int(percent[i]) < 10:
        #     continue
        print("CMYK(" + cmyk_colors[i] + ")", end=" ")


    # ax3 = fig.add_subplot(1, 4, 3)
    # plt.title('HEX', fontsize=10)
    # plt.pie(percentage, labels=hex_colors, colors=hex_colors, autopct="%.0f%%")

    # ax4 = fig.add_subplot(1, 4, 4)
    # plt.title('RGB', fontsize=10)
    # plt.pie(percentage, labels=rgb_colors, colors=hex_colors, autopct="%.0f%%")

    plt.tight_layout()
    plt.show()

# segmentation
def segmented_image(img, centers, labels):
    # convert back to 8 bit values
    centers = np.uint8(centers)
    # print(centers)

    # flatten the labels array
    labels = labels.flatten()
    # print(labels)
    labels = labels
    # print(labels)

    # convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]
    # print(segmented_image)

    # reshape back to the original image dimension
    segmented_image = segmented_image.reshape(img.shape)

    # show the image
    plt.imshow(segmented_image)
    plt.tight_layout()
    plt.axis("off")
    plt.show()
