# USAGE
# python color_kmeans.py --image images/jp.png --clusters 3

# import the necessary packages
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import argparse
import utils
import cv2
import collections
import numpy as np

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
ap.add_argument("-c", "--clusters", required = True, type = int,
	help = "# of clusters")
args = vars(ap.parse_args())

# load the image and convert it from BGR to RGB so that
# we can dispaly it with matplotlib
image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# show our image
plt.figure()
plt.axis("off")
plt.imshow(image)

# reshape the image to be a list of pixels
image = image.reshape((image.shape[0] * image.shape[1], 3))

# cluster the pixel intensities
clt = KMeans(n_clusters = args["clusters"])
clt.fit(image)


# build a histogram of clusters and then create a figure
# representing the number of pixels labeled to each color
hist = utils.centroid_histogram(clt)
# bar = utils.plot_colors(hist, clt.cluster_centers_)

d= {}
for (percent, color) in zip(hist, clt.cluster_centers_):
	p = round(percent,2)
	colors = [int(color[0]), int(color[1]), int(color[2])] # R: color[0], G: color[1], B: color[2]
	d[p] = colors

od = collections.OrderedDict(sorted(d.items(), reverse=True))
print(od)
count = 1
for percent in od:
	if count > 3: break
	color = od[percent]
	# suppose white or black is background
	if (color[0] < 5 and color[1] < 5 and color[2] < 5) or (color[0] > 250 and color[1] > 250 and color[2] > 250):
		print("background")
		continue
	print(str(count) + ": " + "R ("+str(color[0])+"), G ("+str(color[1])+"), B ("+str(color[2])+")")
	count+=1

# show our color bar
hist_len = 300
bar = np.zeros((50, hist_len, 3), dtype = "uint8")
startX = 0
for percent in od:
	color = np.array(od[percent])
	# print(color.astype("uint8").tolist())
	endX = startX + (percent * hist_len)
	cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
		color.astype("uint8").tolist(), -1)
	startX = endX

plt.figure()
plt.axis("off")
plt.imshow(bar)
plt.show()

