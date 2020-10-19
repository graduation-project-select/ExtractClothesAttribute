# USAGE
# Test texture, fabric model
# python classifyTestSingleLabel.py --model model_texture/texture.model --labelbin model_texture/lb.pickle --directory test_data/texture_test
# python classifyTestSingleLabel.py --model model_category/category.model --labelbin model_category/lb.pickle --directory test_data/category_test_single


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

count = "13"

f = open("texture_test"+count+".txt", "w")
fs = open("texture_test"+count+"_simple.txt", "w")
fileContents = ""
fileContents += str(datetime.datetime.now()) + "\n"

fs_fileContents = ""
fs_fileContents += str(datetime.datetime.now()) + "\n"

root_dir = args["directory"]
root_dir_list = os.listdir(root_dir)
for prediction_label in root_dir_list:
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
        image = cv2.imread(img_path)
        # pre-process the image for classification
        image = cv2.resize(image, (96, 96))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        # classify the input image
        # print("["+img+"] classifying image: ")
        fileContents += "["+img+"] classifying image: \n"
        proba = model.predict(image)[0]
        idx = np.argmax(proba)
        label = lb.classes_[idx]
        correct = "correct" if prediction_label.rfind(label) != -1 else "incorrect"
        if correct == "correct" : correct_cnt += 1
        label = "{}: {:.2f}% ({})".format(label, proba[idx] * 100, correct)
        # print(label)
        fileContents += label+"\n"
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
