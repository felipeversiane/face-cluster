from imutils import paths
import face_recognition
import pickle
import cv2
from config import ENCODINGS_PATH, FACE_DATA_PATH
from logger import logger

def validate_image(imagePath):
    image = cv2.imread(imagePath)
    if image is None:
        logger.error(f"Unable to load image: {imagePath}")
        return None
    return image


def is_face_full(face_box, image_shape):
    top, right, bottom, left = face_box
    height, width = image_shape[:2]

    margin = 10 
    if top < margin or left < margin or bottom > height - margin or right > width - margin:
        return False
    
    face_width = right - left
    face_height = bottom - top
    if face_width < 50 or face_height < 50: 
        return False

    center_x, center_y = width // 2, height // 2
    face_center_x = (left + right) // 2
    face_center_y = (top + bottom) // 2

    if abs(center_x - face_center_x) > 0.2 * width or abs(center_y - face_center_y) > 0.2 * height:
        return False

    return True


def process_image(imagePath):
    image = validate_image(imagePath)
    if image is None:
        return None, None

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgb, model="cnn")
    
    if len(boxes) == 0:  
        logger.warning(f"No faces detected in image {imagePath}")
        return None, None
    
    valid_encodings = []
    valid_boxes = []

    for box in boxes:
        if is_face_full(box, image.shape): 
            encodings = face_recognition.face_encodings(rgb, [box])
            if len(encodings) > 0:
                valid_encodings.append(encodings[0])
                valid_boxes.append(box)

    if valid_encodings:
        return valid_boxes[0], valid_encodings[0] 
    else:
        return None, None


logger.info("Quantifying faces...")
imagePaths = list(paths.list_images(FACE_DATA_PATH))
data = []

if not imagePaths:
    logger.error("No images found in the dataset directory.")
    exit(1)

for (i, imagePath) in enumerate(imagePaths):
    logger.info(f"Processing image {i + 1}/{len(imagePaths)}: {imagePath}")
    
    box, encoding = process_image(imagePath)
    if box is None or encoding is None:
        continue

    d = {"imagePath": imagePath, "loc": box, "encoding": encoding}
    data.append(d)

if not data:
    logger.error("No faces found in the dataset.")
    exit(1)

logger.info("Serializing encodings...")
with open(ENCODINGS_PATH, "wb") as f:
    pickle.dump(data, f)

logger.info(f"Encodings saved to {ENCODINGS_PATH}")
