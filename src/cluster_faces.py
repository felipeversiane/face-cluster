# import the necessary packages
from sklearn.cluster import DBSCAN
from imutils import build_montages
import numpy as np
import argparse
import pickle
import cv2
import shutil
import os
from constants import FACE_DATA_PATH, ENCODINGS_PATH, CLUSTERING_RESULT_PATH

# add constants file in the code (clustering_result)

def move_image(image, id, labelID):
    path = os.path.join(CLUSTERING_RESULT_PATH, 'label' + str(labelID))
    if not os.path.exists(path):
        os.mkdir(path)

    filename = str(id) + '.jpg'
    cv2.imwrite(os.path.join(path, filename), image)

    return

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
    help="path to serialized db of facial encodings")
ap.add_argument("-j", "--jobs", type=int, default=-1,
    help="# of parallel jobs to run (-1 will use all CPUs)")
args = vars(ap.parse_args())

# Verificando a existência do arquivo de encodings
if not os.path.exists(args["encodings"]):
    raise ValueError(f"Arquivo {args['encodings']} não encontrado.")

# load the serialized face encodings + bounding box locations from disk
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())
data = np.array(data)
encodings = [d["encoding"] for d in data]

# cluster the embeddings
print("[INFO] clustering...")

# creating DBSCAN object for clustering the encodings with the metric "euclidean"
clt = DBSCAN(metric="euclidean", n_jobs=args["jobs"])
clt.fit(encodings)

# determine the total number of unique faces found in the dataset
labelIDs = np.unique(clt.labels_)
numUniqueFaces = len(np.where(labelIDs > -1)[0])
print(f"[INFO] # unique faces: {numUniqueFaces}")

# loop over the unique face integers
for labelID in labelIDs:
    print(f"[INFO] faces for face ID: {labelID}")
    idxs = np.where(clt.labels_ == labelID)[0]
    idxs = np.random.choice(idxs, size=min(25, len(idxs)), replace=False)

    faces = []

    for i in idxs:
        # load the input image and extract the face ROI
        image = cv2.imread(data[i]["imagePath"])
        (top, right, bottom, left) = data[i]["loc"]
        face = image[top:bottom, left:right]

        move_image(image, i, labelID)

        # resize the face ROI to 96x96 and add it to faces list
        face = cv2.resize(face, (96, 96))
        faces.append(face)

    # create a montage using 96x96 "tiles" with 5 rows and 5 columns
    montage = build_montages(faces, (96, 96), (5, 5))[0]

    title = f"Face ID #{labelID}"
    title = "Unknown Faces" if labelID == -1 else title
    cv2.imwrite(os.path.join(CLUSTERING_RESULT_PATH, f"{title}.jpg"), montage)
