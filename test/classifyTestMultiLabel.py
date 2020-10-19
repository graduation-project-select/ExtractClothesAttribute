# USAGE
# Test texture, fabric model
# python classifyTestMultiLabel.py --model model_category/multiCategory.model --labelbin model_category/mlb.pickle --directory test_data/category_test

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
import datetime
 
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
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-l", "--labelbin", required=True,
	help="path to label binarizer")
ap.add_argument("-i", "--directory", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# load the trained convolutional neural network and the label
# binarizer
print("[INFO] loading network...")
model = load_model(args["model"])
lb = pickle.loads(open(args["labelbin"], "rb").read())

count = "10"

test_type = "category"
f = open(test_type+"_test"+count+".txt", "w")
fs = open(test_type+"_test"+count+"_simple.txt", "w")
fileContents = ""
fileContents += str(datetime.datetime.now()) + "\n"

fs_fileContents = ""
fs_fileContents += str(datetime.datetime.now()) + "\n"


def verifyLabel(prediction_label, image):
    # pre-process the image for classification
    image = cv2.resize(image, (96, 96))
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # grayscale
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    # classify the input image
    proba = model.predict(image)[0]
    # Multi label
    idxs = np.argsort(proba)[::-1][:2]
    # loop over the indexes of the high confidence class labels
    correct = False
    labelInfo = ""
    for (i, j) in enumerate(idxs):
        # build the label and draw the label on the image
        percentage = proba[j] * 100
        resultLabel = "{}: {:.2f}%".format(lb.classes_[j], percentage)
        labelInfo+=resultLabel+"\n"
        if lb.classes_[j] == prediction_label:
            # if (lb.classes_[j] == prediction_label) and (percentage > 50):
            correct = True
            break
    return correct, labelInfo


root_dir = args["directory"]
root_dir_list = os.listdir(root_dir)
for prediction_label in root_dir_list:
    plabel = prediction_label.split("_")[1] # 폴더명이 multi_label인 경우
    fileContents += "===================================\n"
    fileContents += prediction_label+"\n"
    fileContents += "===================================\n"
    print("<"+prediction_label+">")
    fs_fileContents += "<"+prediction_label+">\n"
    sub_dir = root_dir+"/"+prediction_label
    sub_dir_list = os.listdir(sub_dir)
    correct_cnt = 0
    for img in sub_dir_list:
        img_path = sub_dir + "/" + img
        fileContents += "["+img+"] classifying image: \n"
        image = cv2.imread(img_path)
        correct, labelInfo = verifyLabel(plabel, image) # 폴더명이 multi_label인 경우
        # correct, labelInfo = verifyLabel(prediction_label, image) # 폴더명이 하위카테고리(shorts, skirts ...) 인 경우
        fileContents += labelInfo
        if correct:
            fileContents += "(correct)\n"
            correct_cnt += 1
        else:
            fileContents += "(incorrect)\n"
    result_str = ""
    result_str += "correct count: {}/{}".format(correct_cnt, len(sub_dir_list)) + "\n"
    result_str += "accuracy: {:.2f} %".format(correct_cnt/len(sub_dir_list) * 100) + "\n"
    result_str += "-----------------------------------\n"
    print("-----------------------------------")
    print(result_str)
    fileContents += result_str
    fs_fileContents += result_str

f.write(fileContents)
fs.write(fs_fileContents)
f.close()
fs.close()
