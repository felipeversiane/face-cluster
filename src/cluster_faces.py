from sklearn.cluster import AgglomerativeClustering
from imutils import build_montages
import numpy as np
import pickle
import cv2
import os
from config import ENCODINGS_PATH, CLUSTERING_RESULT_PATH
from logger import logger

def move_image(image, id, labelID):
    if labelID == -1:
        return
    path = os.path.join(CLUSTERING_RESULT_PATH, f'label{labelID}')
    os.makedirs(path, exist_ok=True)
    filename = f'{id}.jpg'
    cv2.imwrite(os.path.join(path, filename), image)


def validate_image(imagePath):
    image = cv2.imread(imagePath)
    if image is None:
        logger.error(f"Unable to load image: {imagePath}")
        return None
    return image


logger.info("Loading encodings...")
if not os.path.exists(ENCODINGS_PATH):
    raise ValueError(f"File {ENCODINGS_PATH} not found.")
with open(ENCODINGS_PATH, "rb") as f:
    data = pickle.load(f)

data = np.array(data)
encodings = [d["encoding"] for d in data if "encoding" in d]

if not encodings:
    raise ValueError("No valid encodings found!")

logger.info("Clustering...")
clt = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=0.7,  
    affinity="euclidean",
    linkage="complete",
)
clt.fit(encodings)

labelIDs = np.unique(clt.labels_)
numUniqueFaces = len(labelIDs[labelIDs > -1])
logger.info(f"Unique faces: {numUniqueFaces}")

for labelID in labelIDs:
    logger.info(f"Processing faces for label: {labelID}")
    idxs = np.where(clt.labels_ == labelID)[0]
    if len(idxs) == 0:
        continue

    faces = []
    for i in idxs:
        imagePath = data[i]["imagePath"]
        image = validate_image(imagePath)
        if image is None:
            continue

        (top, right, bottom, left) = data[i]["loc"]
        face = image[int(top):int(bottom), int(left):int(right)]
        face = cv2.resize(face, (96, 96))
        faces.append(face)
        move_image(image, i, labelID)

    if faces:
        montage = build_montages(faces, (96, 96), (5, 5))[0]
        title = f"label{labelID}" if labelID != -1 else "unknown"
        output_path = os.path.join(CLUSTERING_RESULT_PATH, f"{title}.jpg")
        cv2.imwrite(output_path, montage)
    else:
        logger.warning(f"No faces found for label: {labelID}")


