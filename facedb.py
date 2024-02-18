#!/usr/bin/env python3.10
"""
This Python module is a database for storing faces and searching stored faces.

There are two operations:
    - `add <dir or file path to image of face to add>`
    - `search <file path to image of face to search>`

Valid images are: .jpg, .jpeg, .png
This script will use the file "facedb" in its directory as a database or
create it if it does not exist.
"""

import os
import sys
import hashlib
from io import BytesIO
import dbm.ndbm as dbm

import numpy as np
import face_recognition # pip install face_recognition
from numpy.core.multiarray import dtype
import faiss # conda install faiss-cpu -c pytorch

VALID_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

class FaceDB:
    """
    This class wraps the face database.
    In order to add or search faces, one must interact with this class.
    """
    def __init__(self):
        self.db_path = "facedb"
        if os.path.exists(self.db_path):
            self._init_db(self.db_path)
        else:
            self.index = faiss.IndexIDMap2(faiss.IndexFlatL2(128))

    def add_faces(self, images: list[tuple[str,bytes]]) -> list[tuple[str,int]]:
        """
        Add faces to the database.
        Parameters:
            images: list of tuples of (str, bytes) where the str is the abspath to the image and the bytes is the image data.
        Returns:
            list of int: the unique ids of the faces added to the database in the same order as the input images.
        """
        face_encodings = []
        for path, image in images:
            face_encodings_per_image = face_recognition.face_encodings(
                face_recognition.load_image_file(BytesIO(image)), # load image from bytes
                model="large",
                num_jitters=10
            )
            print(f"Found {len(face_encodings_per_image)} faces in image {path}.")
            for iteration, face_encoding in enumerate(face_encodings_per_image):
                face_specific_id = self._get_random_unique_int64() # unique id for each face
                face_encoding = face_encoding.reshape(1, -1) # reshape to 2D array, I don't know why this is necessary
                face_encodings.append((path, face_specific_id, face_encoding))
        for path, face_specific_id, face_encoding in face_encodings:
            # actual add to db
            self.index.add_with_ids(face_encoding, face_specific_id)
            print(f"Added face with id {face_specific_id} to database.")
            self._write_new_state_to_disk()

        return [(path, face_specific_id) for path, face_specific_id, face_encoding in face_encodings]

    def search_face(
        self,
        image_path: str,
        num_results: int = 20,
    ) -> dict[str, int]:
        """
        Search for a face in the database.
        Parameters:
            image_path: str: the abspath to the image to search.
            num_results: int: the number of results to return.
        Returns:
            dict: a dictionary with the file path as the key and the distance as the value.
        """
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image, model="large", num_jitters=10)

        if not face_encodings:
            print(f"No face found in image {image_path}.")
            return []

        print(f"Found {len(face_encodings)} faces in image {image_path}.")

        face_encoding = face_encodings[0].reshape(1, -1) # reshape to 2D array, I don't know why this is necessary
        distances, indices = self.index.search(face_encoding, num_results)
        #print(f"DEBUG: distances: {distances}. indices: {indices}.")
        #print(f"Found {len(indices[0])} results.")

        # get the file paths from ids in facemap
        results = {}
        with dbm.open("facemap", "r") as db:
            # show all key value pairs in facemap dbm db
            """
            for key in db.keys():
                print(f"DEBUG: key: {key}. value: {db[key]}.")
            """
            print(f"DEBUG: indices[0]: {indices[0]}.")
            for iteration, index in enumerate(indices[0]):
                try:
                    # get value (which is the file path) from key (which is the id) in facemap
                    key = str(index).encode("utf-8")
                    results[db[key].decode("utf-8")] = distances[0][iteration]
                except KeyError:
                    print(f"Face with id {index} not found in facemap.")

        return results

    def _init_db(self, db_path: str):
        self.index = faiss.read_index(db_path)

    def _write_new_state_to_disk(self):
        faiss.write_index(self.index, self.db_path)

    def _is_id_in_facemap(self, face_id: int) -> bool:
        with dbm.open("facemap", "c") as db:
            return b'str(face_id)' in db

    def _get_random_unique_int64(self) -> int:
        while True:
            candidate = np.random.randint(0, 2**63)
            if not self._is_id_in_facemap(candidate):
                return candidate

def load_valid_images_from_path(
    path: str,
) -> list[tuple[str, bytes]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image path {path} does not exist.")
    if os.path.isdir(path):
        valid_images_paths = [ os.path.join(path, x) for x in os.listdir(path) if x.endswith(tuple(VALID_IMAGE_EXTENSIONS))]
        if not valid_images_paths:
            raise ValueError(f"No valid images found in directory {path}.")
        valid_images = []
        for img in valid_images_paths:
            image = open(img, "rb").read()
            abs_path = os.path.abspath(img)
            valid_images.append((abs_path, image))
        return valid_images

    if os.path.isfile(path):
        if not path.endswith(tuple(VALID_IMAGE_EXTENSIONS)):
            raise ValueError(f"Image path {path} is not a valid image.")
        abs_path = os.path.abspath(path)
        valid_images = [
            (abs_path,
            open(path, "rb").read()),
        ]
        print(f"Found valid image {path}.")
        return valid_images

    raise ValueError(f"Path {path} is not a dir nor file.")

def add_file_refs_to_kv_store(
    paths: list[str],
    ids: list[int],
):
    """
    Implements dbm to create or append to a mapping of file paths to face ids.
    """
    assert len(paths) == len(ids)
    with dbm.open("facemap", "c") as db:
        for path, face_id in zip(paths, ids):
            db[bytes(str(face_id), 'utf-8')] = bytes(os.path.abspath(path), 'utf-8')
            print(f"Added {face_id} to kv store with path {path}.")


if __name__ == "__main__":
    # arg parsing
    if len(sys.argv) != 3 or sys.argv[1] not in ["add", "search"]:
        print("You did not provide enough arguments. See docs below.")
        print(__doc__)
        print("Checking database")
        FaceDB()
        print("Database check complete.")
        sys.exit(1)
    if sys.argv[1] == "add":
        images = load_valid_images_from_path(sys.argv[2])
        # chunk if too many images (> 1000 in 100 chunks)
        chunks = [images[i:i + 100] for i in range(0, len(images), 100)] if len(images) > 1000 else [images]
        print(f"Adding {len(images)} images in {len(chunks)} chunks.")
        for chunk in chunks:
            paths_ids_tuple = FaceDB().add_faces(chunk)
            add_file_refs_to_kv_store(
                paths=[path for path, _ in paths_ids_tuple],
                ids=[face_id for _, face_id in paths_ids_tuple],
            )
        sys.exit(0)
    elif sys.argv[1] == "search":
        results = FaceDB().search_face(sys.argv[2])
        # print results nicely
        print("Distance | File")
        for path, distance in results.items():
            print(f"{distance:.6f} | open {path}")
        sys.exit(0)
