# USAGE
# python classify.py --model model_category/category.model --labelbin model_category/lb.pickle --image examples/.png

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


class Classification:
    def __init__(self):
        self.result = ""

    @staticmethod
    def setGPU():
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

    @staticmethod
    def imagePreprocessing(imagePath):
        # load the image
        image = cv2.imread(imagePath)
        originalImage = image.copy()

        # pre-process the image for classification
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # gray scale (모델 train 시 전처리에 사용) 추가함
        image = cv2.resize(image, (96, 96))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        return image, originalImage

    @staticmethod
    def loadModel(modelPath, labelbinPath):
        # load the trained convolutional neural network and the label
        # binarizer
        print("[INFO] loading network...")
        model = load_model(modelPath)
        lb = pickle.loads(open(labelbinPath, "rb").read())
        return model, lb

    @staticmethod
    def printAllLabels(lb, proba):
        # 라벨 전체 출력
        for i in range(0, len(lb.classes_)):
            lb_test = "{}: {:.2f}% ".format(lb.classes_[i], proba[i] * 100)
            print(lb_test)

    @staticmethod
    def classify(image, model, lb, output=[], showResult=False):
        # classify the input image
        print("[INFO] classifying image...")
        proba = model.predict(image)[0]
        idx = np.argmax(proba)
        label = lb.classes_[idx]
        result = label
        # printAllLabels(lb, proba)

        if showResult:
            # we'll mark our prediction as "correct" of the input image filename
            # contains the predicted label text (obviously this makes the
            # assumption that you have named your testing image files this way)
            filename = args["image"][args["image"].rfind(os.path.sep) + 1:]
            correct = "correct" if filename.rfind(label) != -1 else "incorrect"

            # build the label and draw the label on the image
            label = "{}: {:.2f}% ({})".format(label, proba[idx] * 100, correct)
            # show the output image
            print("[INFO] {}".format(label))

            if len(output) != 0:
                output = imutils.resize(output, width=400)
                cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)
                cv2.imshow("Output", output)
                cv2.waitKey(0)

        return result


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to trained model model")
ap.add_argument("-l", "--labelbin", required=True,
                help="path to label binarizer")
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
args = vars(ap.parse_args())

Classification.setGPU()
model, lb = Classification.loadModel(args["model"], args["labelbin"])

# image, originalImage = Classification.imagePreprocessing(args["image"])
# result = Classification.classify(image, model, lb)
# print(result)

#######################################

# 모델로 이미지 데이터 분류(필터링 역할)
result_root_dir = "./codibook_datas_"
root_dir = args["image"]
root_dir_list = os.listdir(root_dir)

for prediction_label in root_dir_list:
    sub_dir = root_dir + "/" + prediction_label
    sub_dir_list = os.listdir(sub_dir)
    for img in sub_dir_list:
        img_path = sub_dir + "/" + img
        image, originalImage = Classification.imagePreprocessing(img_path)
        result = Classification.classify(image, model, lb)
        save_path = result_root_dir + "/" + result
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        cv2.imwrite(save_path + "/" + img, originalImage)
        print(result)
