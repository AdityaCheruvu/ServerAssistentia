import os
from shutil import rmtree
from globalVals import *
import subprocess
import face_recognition
import pickle
import cv2
from matplotlib import pyplot as plt

def cropFaceData(pictureDat, locationOfFace):
    lof = locationOfFace
    print(locationOfFace)
    x, y, w, h = locationOfFace
    return pictureDat[y : y+w, x : x+h]

def recognizePeople(data, imgPath):
    image = cv2.imread(imgPath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print("[INFO] recognizing faces...")
    boxes = face_recognition.face_locations(rgb, model="hog")
    blurriness = [(cv2.Laplacian(cropFaceData(image, box),cv2.CV_64F).var()) for box in boxes]
    encodings = face_recognition.face_encodings(rgb, boxes, 1)
    names = {}
    for encoding, blurryVal in zip(encodings, blurriness):
        matches = face_recognition.compare_faces(data["encodings"], encoding, 0.5)
        name = "Unknown"
        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
            name = max(counts, key=counts.get)
        names[name] = blurryVal
    return names, len(boxes), len(names.keys())
       
if __name__ == "__main__":
    try:
        rmtree(tmpDirLocal)
    except:
        pass
    os.mkdir(tmpDirLocal)
    commandStartSystem = 'sshpass -p ' + '"' + raspiPass + '" ' + "ssh " + raspiUser + "@" + raspiIP + " python3.5 /home/pi/Assistentia/RaspberryPiCode/main.py"  
    commandGetImgs = 'sshpass -p ' + '"' + raspiPass + '" ' + "scp " + raspiUser + "@" + raspiIP + ":" + tmpDirRpi + "* " + tmpDirLocal   
    process = subprocess.Popen(commandStartSystem, shell=True, stdout=subprocess.PIPE)
    process.wait()
    print(process.returncode)
    process = subprocess.Popen(commandGetImgs, shell=True, stdout=subprocess.PIPE)
    process.wait()
    print(process.returncode)

    classToMarkAttendance = "4CSE4/"
    print("[INFO] loading encodings...")
    data = pickle.loads(open(classDir+classToMarkAttendance+pickleName, "rb").read())
    for i in os.listdir(tmpDirLocal):
        print(i + " Results: " )
        result = recognizePeople(data, tmpDirLocal+i)
        print(result)