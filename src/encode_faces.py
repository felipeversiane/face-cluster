# encode_faces.py

from imutils import paths
import face_recognition
import pickle
import cv2
import os
from constants import ENCODINGS_PATH, FACE_DATA_PATH

print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(FACE_DATA_PATH))
data = []

for (i, imagePath) in enumerate(imagePaths):
    print(f"[INFO] processing image {i + 1}/{len(imagePaths)}")
    print(imagePath)

    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgb, model="cnn")
    encodings = face_recognition.face_encodings(rgb, boxes)

    for (box, enc) in zip(boxes, encodings):
        d = {"imagePath": imagePath, "loc": box, "encoding": enc}
        data.append(d)

print("[INFO] serializing encodings...")
with open(ENCODINGS_PATH, "wb") as f:
    f.write(pickle.dumps(data))

print(f"Encodings of images saved in {ENCODINGS_PATH}")
