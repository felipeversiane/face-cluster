import os

FACE_DATA_PATH = os.getenv('FACE_DATA_PATH', os.path.join(os.getcwd(), 'face_cluster'))
ENCODINGS_PATH = os.getenv('ENCODINGS_PATH', os.path.join(os.getcwd(), 'encodings.pickle'))
CLUSTERING_RESULT_PATH = os.getenv('CLUSTERING_RESULT_PATH', os.getcwd())

if not os.path.exists(FACE_DATA_PATH):
    os.makedirs(FACE_DATA_PATH)

if not os.path.exists(CLUSTERING_RESULT_PATH):
    os.makedirs(CLUSTERING_RESULT_PATH)

print(f"FACE_DATA_PATH: {FACE_DATA_PATH}")
print(f"ENCODINGS_PATH: {ENCODINGS_PATH}")
print(f"CLUSTERING_RESULT_PATH: {CLUSTERING_RESULT_PATH}")
