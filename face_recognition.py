#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import os
from pathlib import Path
import numpy as np
from random import sample
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
Rigels_TEST = "rigels_test.npy"
Rigels_TRAIN = "rigels_train.npy"
Label_TEST = "label_test.npy"
Label_TRAIN = "label_train.npy"

def detect_face(image):
    if(image is None):
        return None, None
    """Returns a tuple containing (image cropped to face, the matrix representation of it)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    if len(faces) == 0:
        return None, None
    (x, y, w, h) = faces[0]
    return gray[y:y+w, x:x+h], faces[0]


def extract_faces_and_labels(dataset_dir):
    """Returns two lists: (faces, labels) starting from images in the dataset directory"""
    faces = []
    labels = []

    count = 0
    for name in os.listdir(dataset_dir):
        if(name == ".DS_Store"):
            continue
        count += 1
        for image in os.listdir(Path.joinpath(dataset_dir, name)):
            filename = Path.joinpath(dataset_dir, name, image)
            im = cv2.imread(str(filename))
            face, rect = detect_face(im)
            if face is not None:
                faces += [face]
                labels += [count]
                print(f"{count}. Elaborato {image} per celebrity {name}")
    return faces, labels

def prepare_data_for_training_celeb():
    train_dir = Path("celebrity/Celebrity/train")
    test_dir = Path("celebrity/Celebrity/test")

    faces_train, labels_train = extract_faces_and_labels(train_dir)
    faces_test, labels_test = extract_faces_and_labels(test_dir)
    return faces_train, labels_train, faces_test, labels_test


def train_models(faces, labels, force_lbph=True, do_lbph=True):
    labels = np.array(labels)

    if "models" not in os.listdir():
        os.mkdir("models")

    lbph_face_recognizer = cv2.face.LBPHFaceRecognizer_create(radius=3)
    if do_lbph:
        if "LBPH.yml" not in os.listdir("models") or force_lbph:
            lbph_face_recognizer.train(faces, labels)
            lbph_face_recognizer.save("models/LBPH.yml")
        else:
            lbph_face_recognizer.read("models/LBPH.yml")


    return lbph_face_recognizer


def predict(face_recognizer, test_image):
    label, confidence = face_recognizer.predict(test_image)
    return label, confidence


def get_names(dir):
    
    names_dir = Path.joinpath(ROOT, "celebrity/Celebrity/train/")

    return [x.name for x in names_dir.iterdir() if x.name != ".DS_Store"]

   
 
def test_accuracy_thresholds(face_recognizer, faces_test, labels_test, directory):
    statements = []
    thresholds = []
    false_accept_values = []
    false_reject_values = []
    true_accept_values = []
    true_reject_values = []
    true_positive_rates = []
    false_positive_rates = []
    false_reject_rate = []
    accuracies = []
    #la threshold va fino a 201 perché cosi sono i valori delle distanze che restituisce il predict
    
    maxThreshold = 201
    step = 1
    
    for threshold in range(1, maxThreshold, step):
        true_accept = 0
        false_accept = 0
        true_reject = 0
        false_reject = 0
        total = len(faces_test)
        names = get_names(directory)
        for index, face in enumerate(faces_test):
            guess, distance = predict(face_recognizer, face)
            if names[labels_test[index] - 1] != "ZZZ":
                # Here we can have true accept or false reject
                if distance > threshold:
                    false_reject += 1
                else:
                    true_accept += 1
            else:
                # Here we can have false accept or true reject
                if distance > threshold:
                    true_reject += 1
                else:
                    false_accept += 1
        statements += [f"With threshold: {threshold}: TR: {true_reject}, TA: {true_accept}, FR: {false_reject}, FA: {false_accept}, accuracy: {(true_accept + true_reject) / total}"]
        thresholds += [threshold]
        true_reject_values += [true_reject]
        true_accept_values += [true_accept]
        false_reject_values += [false_reject]
        false_accept_values += [false_accept]
        accuracies += [(true_accept + true_reject) / total]
        tpr = true_accept/(true_accept + false_reject)
        fpr = false_accept/(false_accept + true_reject)
        frr = false_reject/(true_accept + false_reject)
        false_reject_rate += [frr]
        true_positive_rates += [tpr]
        false_positive_rates += [fpr]
        print("Threshold is " + str(threshold) + ", " + "fpr is: " + fpr + ", " + "frr is: " + frr)
    
    plt.plot(false_positive_rates, true_positive_rates)
    plt.show()


if __name__ == '__main__':
    
    faces_train, labels_train, faces_test, labels_test = prepare_data_for_training_celeb()
    
    lbph_face_recognizer = train_models(faces_train, labels_train)
    
    image = cv2.imread("/Users/rigelshysaj/progetti/celebrity/Celebrity/test/Obama/iui.jpg")
    
    face, rect = detect_face(image)
    
    label, confidence = predict(lbph_face_recognizer, face)
    
    names = get_names("celeb")
    
    print(names)
    
    print(names[label-1])
    
    
