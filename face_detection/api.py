from __future__ import print_function
import os
import face_recognition
import cv2
import torch
from torch.utils.model_zoo import load_url
from enum import Enum
import numpy as np
import cv2
try:
    import urllib.request as request_file
except BaseException:
    import urllib as request_file

from .models import FAN, ResNetDepth
from .utils import *


def VideoEngine(KNOWN_FACE, UNKNOWN_FACE,defAdj = 10):
    knownFace = KNOWN_FACE
    unknownFace = UNKNOWN_FACE
    TOLERANCE = 0.8
    MODEL = 'cnn'
    video = cv2.VideoCapture(UNKNOWN_FACE)
    bboxlists = []
    known_faces = []
    known_names = []
    image = face_recognition.load_image_file(knownFace)
    encoding = face_recognition.face_encodings(image)[0]
    #Appending names and encodings
    known_faces.append(encoding)
    known_names.append('0')
    while True:
        try:
            ret, image = video.read()
            locations = face_recognition.face_locations(image,model=MODEL)
            encodings = face_recognition.face_encodings(image,locations)
            for face_encoding,face_location in zip(encodings, locations):
                results = face_recognition.compare_faces(known_faces, face_encoding,TOLERANCE)
                match = None
                if True in results:
                    match = known_names[results.index(True)]
                    top_left = (face_location[3], face_location[0])# for future dev
                    bottom_right = (face_location[1], face_location[2])
                    x1 = face_location[3]
                    y1 = face_location[0]
                    x2 = face_location[1]
                    y2 = face_location[2]
                    bboxlist = [x1-defAdj,y1-defAdj,x2-defAdj,y2-defAdj]
                    bboxlists.append(bboxlist)
                else:
                    bboxlists.append([0,0,0,0])
        except TypeError:
            break
    return bboxlists


def ImageEngine(knownFace, unknownFace,defAdj = 10):
    KNOWN_FACE = knownFace
    UNKNOWN_FACE = unknownFace
    TOLERANCE = 0.8
    MODEL = 'cnn'
    known_faces = []
    known_names = []
    bboxlists = []
    
    image = face_recognition.load_image_file(KNOWN_FACE)
    encoding = face_recognition.face_encodings(image)[0]
    known_faces.append(encoding)
    known_names.append('0')
    UKimage = face_recognition.load_image_file(UNKNOWN_FACE)
    locations = face_recognition.face_locations(UKimage,model = MODEL)
    encodings = face_recognition.face_encodings(UKimage,locations)
    for face_encoding, face_location in zip(encodings, locations):
        result = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        match = None
        if True in results:
            match = known_names[results.index(True)]
            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1],face_location[2])
            x1 = face_location[3]
            y1 = face_location[0]
            x2 = face_location[1]
            y2 = face_location[2]
            bboxlist = [x1-defAdj,y1-defAdj, x2-defAdj, y2-defAdj]
            bboxlists.append(bboxlist)
    return bboxlists


def get_detections_for_batch(KNOWN_FACE, UNKNOWN_FACE,defAdj = 10):
    detected_faces = VideoEngine(KNOWN_FACE, UNKNOWN_FACE,defAdj = 10)
    results = []
    for i in range(0,len(detected_faces)):
        x1,y1,x2,y2 = map(int,detected_faces[i])
        results.append((x1,y1,x2,y2))
        return results
                    
def get_detections_for_image(KNOWN_FACE, UNKNOWN_FACE,defAdj = 10):
    detected_faces = ImageEngine(KNOWN_FACE, UNKNOWN_FACE,defAdj = 10)
    results = []
    for i in range(0,len(detected_faces)):
        x1,y1,x2,y2 = map(int,detected_faces[i])
        results.append((x1,y1,x2,y2))
        return results
    
        
        
    
    