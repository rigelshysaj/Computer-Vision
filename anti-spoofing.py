import torch
import modelsCV
import cv2
import argparse
import numpy as np
import imutils
import time
from imutils.video import VideoStream
from datetime import datetime
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
import face_recognition as fr


ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

protoPath = "./face_detector/deploy.prototxt"
modelPath = "./face_detector/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

protoPath2 = "./face_alignment/2_deploy.prototxt"
modelPath2 = "./face_alignment/2_solver_iter_800000.caffemodel"
net2 = cv2.dnn.readNetFromCaffe(protoPath2, modelPath2)

model_name = "MyresNet18"
load_model_path = "a8.pth"
model = getattr(modelsCV, model_name)().eval()
model.load(load_model_path)
model.train(False)

ATTACK = 1
GENUINE = 0
thresh = 0.7

print(protoPath)

def detector(img):
    frame = imutils.resize(img, width=600)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > args["confidence"]:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            face = frame[startY:endY, startX:endX]
            face = cv2.resize(face, (400, 400))
            return face

def crop_with_ldmk(image, landmark):
    scale = 3.5
    image_size = 224
    ct_x, std_x = landmark[:, 0].mean(), landmark[:, 0].std()
    ct_y, std_y = landmark[:, 1].mean(), landmark[:, 1].std()

    std_x, std_y = scale * std_x, scale * std_y

    src = np.float32([(ct_x, ct_y), (ct_x + std_x, ct_y + std_y), (ct_x + std_x, ct_y)])
    dst = np.float32([((image_size - 1) / 2.0, (image_size - 1) / 2.0),
                      ((image_size - 1), (image_size - 1)),
                      ((image_size - 1), (image_size - 1) / 2.0)])
    retval = cv2.getAffineTransform(src, dst)
    result = cv2.warpAffine(image, retval, (image_size, image_size), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT)
    return result

def demo(img):
    data= np.transpose(np.array(img, dtype=np.float32), (2, 0, 1))
    data = data[np.newaxis, :]
    data = torch.FloatTensor(data)
    with torch.no_grad():
        outputs = model(data)
        outputs = torch.softmax(outputs, dim=-1)
        preds = outputs.to('cpu').numpy()
        attack_prob = preds[:, ATTACK]
    return  attack_prob




def faceAndEyeDetection():
    #Face and eye cascade classifiers from xml files
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')
    
    first_read = True
    eyeDetection = False
    blinkDetection = False
    # Video Capturing by using webcam
    #cap = cv2.VideoCapture(0)
    vs = VideoStream(src=0).start()
    #ret, image = cap.read()
    inizioTempo = datetime.now()
    
    
    while True:
        # this will keep the web-cam running and capturing the image for every loop
        image = vs.read()
        
        returnImage = image.copy() 
        # Convert the rgb image to gray
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Applying bilateral filters to remove impurities
        gray = cv2.bilateralFilter(gray, 5, 1, 1)
        # to detect face 
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(200, 200))
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                image = cv2.rectangle(image, (x, y), (x + w, y + h), (1, 190, 200), 2)
                #face detector
                roi_face = gray[y:y + h, x:x + w]
                # image
                roi_face_clr = image[y:y + h, x:x + w]
                # to detect eyes
                eyes = eye_cascade.detectMultiScale(roi_face, 1.3, 5, minSize=(50, 50))
                for (ex,ey,ew,eh) in eyes:
                        cv2.rectangle(roi_face_clr,(ex,ey),(ex+ew,ey+eh),(255, 153, 255),2)
                        if len(eyes) >= 2:
                            if first_read:
                                cv2.putText(image, "Eye's detected", (70, 70), cv2.FONT_HERSHEY_TRIPLEX,
                                    1, (255, 0, 0), 2)
                                eyeDetection = True
                                
                            else:
                                cv2.putText(image, "Eye's Open", (70, 70), cv2.FONT_HERSHEY_TRIPLEX,
                                    1, (255, 255, 255), 2)
                        else:
                            if first_read:
                                cv2.putText(image, "No Eye's detected", (70, 70), cv2.FONT_HERSHEY_TRIPLEX,
                                    1, (255, 0, 255), 2)
                            else:
                                cv2.putText(image, "Blink Detected.....!!!!", (70, 70), cv2.FONT_HERSHEY_TRIPLEX,
                                    1, (0, 0, 0), 2)
                                print("Blink Detected.....!!!!")
                                blinkDetection = True
                                
        else:
            cv2.putText(image, "No Face Detected.", (70, 70),cv2.FONT_HERSHEY_TRIPLEX,
                        1, (0, 255, 255), 2)
        cv2.imshow('Frame', image)
        cv2.waitKey(1)
        
        fineTempo = datetime.now() - inizioTempo
        
        if(fineTempo.seconds > 5 and eyeDetection):
            first_read = False
        if(fineTempo.seconds > 9 and blinkDetection and eyeDetection):
            break
        
    
    if not blinkDetection:
        return "No face or blink detected"
    
    return returnImage, vs




def spoofing_liveness():
    
    faces_train, labels_train, faces_test, labels_test = fr.prepare_data_for_training_celeb()

    lbph_face_recognizer, eigen_face_recognizer, fisher_face_recognizer = fr.train_models(faces_train, labels_train)
    
    frame, vs = faceAndEyeDetection()
    
    
    if(not isinstance(frame, str)):
        print("[INFO] starting video stream...")
        time.sleep(2.0)
    
        while True:
            frame = vs.read() #the type of the frame is numpy, the shape is (720, 1280, 3)
            
            #frame = imutils.resize(frame, width=600) se da errore o non funziona il funzionamento totale allora è da togliere il commento

            celebr = face_rec(frame, lbph_face_recognizer)            

            spoofing_detection(frame, thresh, celebr)            

            cv2.imshow("Frame", frame)
            cv2.waitKey(1)
        else:
            return frame
    
        cv2.destroyAllWindows()
        vs.stop()
        

def face_rec(image, lbph_face_recognizer):

    face, rect = fr.detect_face(image)

    label, confidence = fr.predict(lbph_face_recognizer, face)

    names = fr.get_names("celeb")
    
    celebr = names[label-1]
    
    return celebr


def spoofing_detection(frame, thresh, celebr = ""):
    (h, w) = frame.shape[:2]
    
    #Example of resize: cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    #Example of dnn.blobFromImage: cv2.dnn.blobFromImage(image, scalefactor=1.0, size, mean, swapRB=True)
    #Since we are declaring size = (300, 300), i think cv2.resize(frame, (300, 300)) is useless, should be only frame
    #The cv2.dnn.blobFromImage function returns a blob which is our input image after mean subtraction, normalizing, and channel swapping.
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
    
    # set the input to the pre-trained deep learning network and obtain
    # the output predicted probabilities for each of the 1,000 ImageNet classes
    net.setInput(blob)
    detections = net.forward()
    
    #print(detections.shape) = (1, 1, 200, 7)
    #npArray = np.array([[[[3,4,5,6,7,8,9], [10,11,12,13,14,15,16], [17,18,19,20,21,22,23], [24,25,26,27,28,29,30]]]]) questa per esempio ha la dimensione = (1, 1, 4, 7)
    #Per arrivare a 200 basta aggiungere altre 193 liste di questo tipo, per esempio, [float, float, floatt, float, float, float, float] dentro l'ultima dimensione, come ho fatto sopra
    #print(npArray[0,0,1,2]) stampa 12
    #The 3rd dimension has our predictions, and each prediction is a list of 7 floating values. At the 1 index we have the class_id, at 2nd index we have the confidence/probability and from 3rd to 6th index we have the coordinates of the object detected.
    
    label = ''
    
    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]
        if confidence > args["confidence"]:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            sx = startX
            sy = startY
            ex = endX
            ey = endY

            ww = (endX - startX) // 10
            hh = (endY - startY) // 5

            startX = startX - ww
            startY = startY + hh
            endX = endX + ww
            endY = endY + hh

            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            x1 = int(startX)
            y1 = int(startY)
            x2 = int(endX)
            y2 = int(endY)

            roi = frame[y1:y2, x1:x2]
            if(roi.size == 0):
                continue
            gary_frame = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            resize_mat = np.float32(gary_frame)
            m = np.zeros((40, 40))
            sd = np.zeros((40, 40))
            mean, std_dev = cv2.meanStdDev(resize_mat, m, sd)
            new_m = mean[0][0]
            new_sd = std_dev[0][0]
            new_frame = (resize_mat - new_m) / (0.000001 + new_sd)
            blob2 = cv2.dnn.blobFromImage(cv2.resize(new_frame, (40, 40)), 1.0, (40, 40), (0, 0, 0))
            net2.setInput(blob2)
            align = net2.forward()

            aligns = []
            alignss = []
            for i in range(0, 68):
                align1 = []
                x = align[0][2 * i] * (x2 - x1) + x1
                y = align[0][2 * i + 1] * (y2 - y1) + y1
                cv2.circle(frame, (int(x), int(y)), 1, (0, 0, 255), 2)
                align1.append(int(x))
                align1.append(int(y))
                aligns.append(align1)
            cv2.rectangle(frame, (sx, sy), (ex, ey),(0, 0, 255), 2)
            alignss.append(aligns)

            ldmk = np.asarray(alignss, dtype=np.float32)
            ldmk = ldmk[np.argsort(np.std(ldmk[:,:,1],axis=1))[-1]]
            img = crop_with_ldmk(frame,ldmk)

            attack_prob = demo(img)
            

            true_prob = 1 - attack_prob
            if attack_prob > thresh:
                label = 'fake ' + celebr
                cv2.putText(frame, label+' :'+str(attack_prob), (sx, sy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                label = 'true ' + celebr
                cv2.putText(frame, label+' :'+str(true_prob), (sx, sy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return label



def test_accuracy_blink():
    true_positive_rates = []
    false_positive_rates = []

    thresholds = []
    true_positive_values = []
    false_negative_values = []
    true_negative_values = []
    false_positive_values = []
    accuracies = []
    
    
    path = "/Users/rigelshysaj/Downloads/LCC_FASD/LCC_FASD_development/"
    
    for i in range(1, 11):
        tn = 0
        tp = 0
        fp = 0
        fn = 0
        thresh = i/10
    
        for directory in os.listdir(path):
            if(directory == ".DS_Store"):
                continue
            for filename in os.listdir(path + directory):
                image = cv2.imread(path + directory + "/" + filename)
                if image is None:
                    continue
                
                image = cv2.resize(image, dsize=(1280, 720))
                
                if(image is None):
                    continue
                
                label = spoofing_detection(image, thresh)
                
                if label is None:
                    continue
    
                if label == 'true':
                    if directory == "real":
                        tp += 1
                    else:
                        fp += 1
                elif label == 'fake':
                    if directory == "spoof":
                        tn += 1
                    else:
                        fn += 1
        print(f"Accuracy at threshold {i / 10}: {(tp + tn) / (tp + tn + fp + fn)}")
        tpr = tp/(tp + fn)
        fpr = fp/(fp + tn)
        true_positive_rates += [tpr]
        false_positive_rates += [fpr]
        thresholds += [i]
        true_positive_values += [tp]
        true_negative_values += [tn]
        false_positive_values += [fp]
        false_negative_values += [fn]
        accuracies += [(tp + tn) / (tp + tn + fp + fn)]
    
    print(f"TP {tp}, TN: {tn}, FP {fp}, FN: {fn}")
    plt.plot(false_positive_rates, true_positive_rates)
    plt.show()


#spoofing_liveness()
