# cluster_faces.py

from sklearn.cluster import DBSCAN
from imutils import build_montages
import numpy as np
import pickle
import cv2
import shutil
import os
from constants import FACE_DATA_PATH, ENCODINGS_PATH, CLUSTERING_RESULT_PATH

def move_image(image, id, labelID):
    path = os.path.join(CLUSTERING_RESULT_PATH, f'label{labelID}')
    if not os.path.exists(path):
        os.makedirs(path)
    filename = f'{id}.jpg'
    cv2.imwrite(os.path.join(path, filename), image)

print("[INFO] loading encodings...")
if not os.path.exists(ENCODINGS_PATH):
    raise ValueError(f"Arquivo {ENCODINGS_PATH} não encontrado.")
with open(ENCODINGS_PATH, "rb") as f:
    data = pickle.load(f)

data = np.array(data)
encodings = [d["encoding"] for d in data]

print("[INFO] clustering...")
clt = DBSCAN(metric="euclidean", n_jobs=-1)
clt.fit(encodings)

labelIDs = np.unique(clt.labels_)
numUniqueFaces = len(np.where(labelIDs > -1)[0])
print(f"[INFO] # unique faces: {numUniqueFaces}")

for labelID in labelIDs:
    print(f"[INFO] faces for face ID: {labelID}")
    idxs = np.where(clt.labels_ == labelID)[0]
    idxs = np.random.choice(idxs, size=min(25, len(idxs)), replace=False)

    faces = []

    for i in idxs:
        image = cv2.imread(data[i]["imagePath"])
        (top, right, bottom, left) = data[i]["loc"]
        face = image[int(top):int(bottom), int(left):int(right)]

        move_image(image, i, labelID)
        face = cv2.resize(face, (96, 96))
        faces.append(face)

    montage = build_montages(faces, (96, 96), (5, 5))[0]

    title = f"Face ID #{labelID}"
    title = "Unknown Faces" if labelID == -1 else title
    cv2.imwrite(os.path.join(CLUSTERING_RESULT_PATH, f"{title}.jpg"), montage)
