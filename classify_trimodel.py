# USAGE
# python classify_trimodel.py --image examples/1.jpg

# import the necessary packages
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

# import for extract color
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import utils
import collections


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-mc", "--modelc", required=True,
# 	help="path to trained category model")
# ap.add_argument("-mt", "--modelt", required=True,
# 	help="path to trained texture model")
# ap.add_argument("-mf", "--modelf", required=True,
# 	help="path to trained fabric model")
# ap.add_argument("-lc", "--labelbinc", required=True,
# 	help="path to label binarizer")
#   ap.add_argument("-lt", "--labelbint", required=True,
# 	help="path to label binarizer")
#   ap.add_argument("-lf", "--labelbinf", required=True,
# 	help="path to label binarizer")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

model_c = "model_category/multiCategory.model"
labelbin_c = "model_category/mlb.pickle"
model_t = "model_texture/texture.model"
labelbin_t = "model_texture/lb.pickle"
model_f = "model_fabric/fabric.model"
labelbin_f = "model_fabric/lb.pickle"

# load the image
image = cv2.imread(args["image"])
image_color = image.copy()
output = image.copy()
 
# pre-process the image for classification
image = cv2.resize(image, (96, 96))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# load the trained convolutional neural network and the label
# binarizer
print("[INFO] loading network...")
print("[INFO] loading network - category...")
modelc = load_model(model_c)
lbc = pickle.loads(open(labelbin_c, "rb").read())
print("[INFO] loading network - texture...")
modelt = load_model(model_t)
lbt = pickle.loads(open(labelbin_t, "rb").read())
print("[INFO] loading network - fabric...")
modelf = load_model(model_f)
lbf = pickle.loads(open(labelbin_f, "rb").read())

# classify the input image
print("[INFO] classifying image...")
print("[INFO] classifying image - category...")
probac = modelc.predict(image)[0]
idxc = np.argmax(probac)
labelc = lbc.classes_[idxc]

# multi label
print("[INFO] classifying image - category...")
probac = modelc.predict(image)[0]
idxcs = np.argsort(probac)[::-1][:2]
# loop over the indexes of the high confidence class labels
for (i, j) in enumerate(idxcs):
  # build the label and draw the label on the image
  label = "{}: {:.2f}%".format(lbc.classes_[j], probac[j] * 100)
  print("multi-label: "+label)
  # cv2.putText(output, label, (10, (i * 30) + 25),
  # 	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

print("[INFO] classifying image - texture...")
probat = modelt.predict(image)[0]
idxt = np.argmax(probat)
labelt = lbt.classes_[idxt]
print("[INFO] classifying image - fabric...")
probaf = modelf.predict(image)[0]
idxf = np.argmax(probaf)
labelf = lbf.classes_[idxf]

# # we'll mark our prediction as "correct" of the input image filename
# # contains the predicted label text (obviously this makes the
# # assumption that you have named your testing image files this way)
# filename = args["image"][args["image"].rfind(os.path.sep) + 1:]
# correct = "correct" if filename.rfind(label) != -1 else "incorrect"

# build the label and draw the label on the image
# labelc = "{}: {:.2f}% ({})".format(labelc, probac[idxc] * 100, correct)
labelc = "{}: {:.2f}% ({})".format(labelc, probac[idxc] * 100, "")
labelt = "{}: {:.2f}% ({})".format(labelt, probat[idxt] * 100, "")
labelf = "{}: {:.2f}% ({})".format(labelf, probaf[idxf] * 100, "")
# output = imutils.resize(output, width=400)
# cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
# 	0.7, (0, 255, 0), 2)

# show the output image
print("[INFO] {}".format(labelc))
print("[INFO] {}".format(labelt))
print("[INFO] {}".format(labelf))
# cv2.imshow("Output", output)
# cv2.waitKey(0)


# extract color
image_color = cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB)
image_color = image_color.reshape((image_color.shape[0] * image_color.shape[1], 3))

clt = KMeans(n_clusters = 3)
clt.fit(image_color)

hist = utils.centroid_histogram(clt)

d = {}
for (percent, color) in zip(hist, clt.cluster_centers_):
  p = round(percent, 2)
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
