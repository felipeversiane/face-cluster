from imutils import paths
import face_recognition
import pickle
import cv2
from constants import ENCODINGS_PATH, FACE_DATA_PATH


def validate_image(imagePath):
    image = cv2.imread(imagePath)
    if image is None:
        print(f"[ERROR] Unable to load image: {imagePath}")
        return None
    return image


print("[INFO] Quantifying faces...")
imagePaths = list(paths.list_images(FACE_DATA_PATH))
data = []

if not imagePaths:
    print("[ERROR] No images found in the dataset directory.")
    exit(1)

for (i, imagePath) in enumerate(imagePaths):
    print(f"[INFO] Processing image {i + 1}/{len(imagePaths)}: {imagePath}")

    image = validate_image(imagePath)
    if image is None:
        continue

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model="cnn")
    encodings = face_recognition.face_encodings(rgb, boxes)

    for (box, enc) in zip(boxes, encodings):
        d = {"imagePath": imagePath, "loc": box, "encoding": enc}
        data.append(d)

if not data:
    print("[ERROR] No faces found in the dataset.")
    exit(1)

print("[INFO] Serializing encodings...")
with open(ENCODINGS_PATH, "wb") as f:
    pickle.dump(data, f)

print(f"Encodings saved to {ENCODINGS_PATH}")
