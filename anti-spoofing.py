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



def spoofing_liveness():
    
    faces_train, labels_train, faces_test, labels_test = fr.prepare_data_for_training_celeb()

    lbph_face_recognizer = fr.train_models(faces_train, labels_train)
    
    vs = VideoStream(src=0).start()
    
    #frame, vs = faceAndEyeDetection()
    
    
    
    print("[INFO] starting video stream...")
    time.sleep(2.0)

    while True:
        frame = vs.read() #the type of the frame is numpy, the shape is (720, 1280, 3)
        
        #frame = imutils.resize(frame, width=600) se da errore o non funziona il funzionamento totale allora è da togliere il commento

        celebr = face_rec(frame, lbph_face_recognizer)

        if(celebr is None):
            continue

        spoofing_detection(frame, thresh, celebr)            

        cv2.imshow("Frame", frame)
        cv2.waitKey(1)
    else:
        return frame

    cv2.destroyAllWindows()
    vs.stop()
        

def face_rec(image, lbph_face_recognizer):

    face, rect = fr.detect_face(image)
    
    if(face is None):
        return None

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



def test_accuracy():
    true_positive_rates = []
    false_positive_rates = []
    precisions = []

    
    
    path = "LCC_FASD/LCC_FASD_development/"
    
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
    
                label = label.strip()
                if label == 'true':
                    if directory == "real":
                        tn += 1
                    else:
                        fn += 1
                elif label == 'fake':
                    if directory == "spoof":
                        tp += 1
                    else:
                        fp += 1
            
        
        precision = tp/(tp + fp)
        tpr = tp/(tp + fn)
        fpr = fp/(fp + tn)
        true_positive_rates += [tpr]
        false_positive_rates += [fpr]
        precisions += [precision]
        
        print(f"Accuracy at threshold {i / 10}: {(tp + tn) / (tp + tn + fp + fn)}")
        print(f"Precision at threshold {i / 10}: {precision}")
        print(f"Recall at threshold {i / 10}: {tpr}")
        print(f"F1 at threshold {i / 10}: {(2 * precision * tpr) / (precision + tpr)}")
    
    
    plt.plot(false_positive_rates, true_positive_rates)
    plt.show()
    plt.plot(true_positive_rates, precisions)
    plt.show()

test_accuracy()
