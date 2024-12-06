from imutils import paths
import face_recognition
import pickle
import cv2
from constants import ENCODINGS_PATH, FACE_DATA_PATH
from logger import logger


def validate_image(imagePath):
    image = cv2.imread(imagePath)
    if image is None:
        logger.error(f"Unable to load image: {imagePath}")
        return None
    return image


logger.info("Scanning dataset for images...")
imagePaths = list(paths.list_images(FACE_DATA_PATH))
data = []

if not imagePaths:
    logger.error("No images found in the dataset directory.")
    exit(1)

for (i, imagePath) in enumerate(imagePaths):
    logger.info(f"Processing image {i + 1}/{len(imagePaths)}: {imagePath}")

    image = validate_image(imagePath)
    if image is None:
        continue

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model="cnn")
    encodings = face_recognition.face_encodings(rgb, boxes)

    if not boxes:
        logger.warning(f"No faces detected in image: {imagePath}")

    for (box, enc) in zip(boxes, encodings):
        d = {"imagePath": imagePath, "loc": box, "encoding": enc}
        data.append(d)

if not data:
    logger.error("No faces detected in any of the images.")
    exit(1)

logger.info("Saving face encodings...")
with open(ENCODINGS_PATH, "wb") as f:
    pickle.dump(data, f)

logger.info(f"Encodings saved successfully at {ENCODINGS_PATH}")