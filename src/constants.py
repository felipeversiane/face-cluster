import os

FACE_DATA_PATH = os.getenv('FACE_DATA_PATH', os.path.join(os.getcwd(), 'database'))
ENCODINGS_PATH = os.getenv('ENCODINGS_PATH', os.path.join(os.getcwd(), 'encodings', 'encodings.pickle'))
CLUSTERING_RESULT_PATH = os.getenv('CLUSTERING_RESULT_PATH', os.path.join(os.getcwd(), 'results'))

if not os.path.exists(FACE_DATA_PATH):
    os.makedirs(FACE_DATA_PATH)

if not os.path.exists(CLUSTERING_RESULT_PATH):
    os.makedirs(CLUSTERING_RESULT_PATH)
