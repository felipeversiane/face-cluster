from sklearn.cluster import AgglomerativeClustering
from imutils import build_montages
import numpy as np
import pickle
import cv2
import os
from config import ENCODINGS_PATH, CLUSTERING_RESULT_PATH
from logger import logger


def validate_image(imagePath):
    """
    Validates and loads an image from the specified path.

    Args:
        imagePath (str): The path to the image.

    Returns:
        numpy.ndarray or None: The loaded image as an array, or None if loading fails.
    """
    image = cv2.imread(imagePath)
    if image is None:
        logger.error(f"Unable to load image: {imagePath}")
        return None
    return image


def move_image(image, image_id, labelID):
    """
    Moves an image to the corresponding clustering result folder.

    Args:
        image (numpy.ndarray): The image to be moved.
        image_id (int): The ID of the image.
        labelID (int): The cluster label ID.
    """
    if labelID == -1:
        return  

    path = os.path.join(CLUSTERING_RESULT_PATH, f'label{labelID}')
    os.makedirs(path, exist_ok=True)

    filename = f'{image_id}.jpg'
    cv2.imwrite(os.path.join(path, filename), image)


def load_encodings(encodings_path):
    """
    Loads face encodings from a file.

    Args:
        encodings_path (str): Path to the serialized encodings file.

    Returns:
        tuple: A tuple containing the encoding data and the corresponding metadata.
    """
    logger.info("Loading encodings...")
    if not os.path.exists(encodings_path):
        raise FileNotFoundError(f"File {encodings_path} not found.")

    with open(encodings_path, "rb") as f:
        data = pickle.load(f)

    data = np.array(data)
    encodings = [d["encoding"] for d in data if "encoding" in d]

    if not encodings:
        raise ValueError("No valid encodings found!")

    return data, encodings


def perform_clustering(encodings, distance_threshold=0.7):
    """
    Performs hierarchical clustering on the given encodings.

    Args:
        encodings (list): List of face encodings.
        distance_threshold (float): Maximum distance threshold for clustering.

    Returns:
        numpy.ndarray: Cluster labels for each encoding.
    """
    logger.info("Clustering...")
    clt = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        affinity="euclidean",
        linkage="complete",
    )
    clt.fit(encodings)
    return clt.labels_


def process_clusters(data, labels):
    """
    Processes clusters and saves the results as montages.

    Args:
        data (numpy.ndarray): Metadata for images (paths and bounding boxes).
        labels (numpy.ndarray): Cluster labels for the encodings.
    """
    labelIDs = np.unique(labels)
    numUniqueFaces = len(labelIDs[labelIDs > -1])
    logger.info(f"Unique faces: {numUniqueFaces}")

    for labelID in labelIDs:
        logger.info(f"Processing faces for label: {labelID}")
        idxs = np.where(labels == labelID)[0]
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


def main():
    """
    Main function to perform face clustering and save results.
    """
    try:
        data, encodings = load_encodings(ENCODINGS_PATH)
        labels = perform_clustering(encodings)
        process_clusters(data, labels)

    except Exception as e:
        logger.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
