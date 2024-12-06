import random
from imutils import paths
import face_recognition
import pickle
import cv2
from config import ENCODINGS_PATH, FACE_DATA_PATH
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


def get_largest_face(boxes):
    """
    Finds the largest face based on the area of the bounding box.

    Args:
        boxes (list): List of face bounding boxes in the format [(top, right, bottom, left), ...].

    Returns:
        tuple or None: The largest bounding box or None if the list is empty.
    """
    if not boxes:
        return None
    return max(boxes, key=lambda box: (box[2] - box[0]) * (box[1] - box[3]))


def process_image(imagePath):
    """
    Processes an image to detect the largest face and generate its encoding.

    Args:
        imagePath (str): The path to the image.

    Returns:
        tuple: The largest bounding box and its encoding, or (None, None) if processing fails.
    """
    image = validate_image(imagePath)
    if image is None:
        return None, None

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect all face bounding boxes
    boxes = face_recognition.face_locations(rgb, model="cnn")

    if len(boxes) == 0:
        logger.warning(f"No faces detected in image {imagePath}")
        return None, None

    # Select the largest face
    largest_box = get_largest_face(boxes)
    if largest_box is None:
        logger.warning(f"No valid faces detected in image {imagePath}")
        return None, None

    # Generate encoding for the largest face
    encodings = face_recognition.face_encodings(rgb, [largest_box])
    if len(encodings) > 0:
        return largest_box, encodings[0]
    else:
        return None, None


def process_images(imagePaths):
    """
    Processes all images to detect and encode faces.

    Args:
        imagePaths (list): List of image paths.

    Returns:
        list: A list of dictionaries with processed face data.
    """
    data = []
    for i, imagePath in enumerate(imagePaths):
        logger.info(f"Processing image {i + 1}/{len(imagePaths)}: {imagePath}")
        box, encoding = process_image(imagePath)
        if box is None or encoding is None:
            continue
        data.append({"imagePath": imagePath, "loc": box, "encoding": encoding})
    return data


def save_encodings(data, output_path):
    """
    Saves face encodings to a file.

    Args:
        data (list): The face encoding data.
        output_path (str): The path to save the file.
    """
    logger.info("Serializing encodings...")
    with open(output_path, "wb") as f:
        pickle.dump(data, f)
    logger.info(f"Encodings saved to {output_path}")


def main():
    """
    Main function to quantify faces and save encodings.
    """
    logger.info("Quantifying faces...")
    imagePaths = list(paths.list_images(FACE_DATA_PATH))
    
    logger.info("Shuffling images...")
    random.shuffle(imagePaths)
    
    if not imagePaths:
        logger.error("No images found in the dataset directory.")
        return

    data = process_images(imagePaths)
    if not data:
        logger.error("No faces found in the dataset.")
        return

    save_encodings(data, ENCODINGS_PATH)


if __name__ == "__main__":
    main()
