import face_recognition
import cv2
import numpy as np


class SimpleFacerec:
    def __init__(self):

        self.user_name = ""
        self.known_face_encodings = []
        self.known_face_names = []

        # Resize frame for a faster speed
        self.frame_resizing = 0.25

    def load_encoding_images(self, name):

        pic = f"Images/{name}.jpg"
        Cus_name = f"{name}"

        img = cv2.imread(pic)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Get encoding
        img_encoding = face_recognition.face_encodings(rgb_img)[0]

        # Store file name and file encoding
        self.known_face_encodings.append(img_encoding)
        self.known_face_names.append(Cus_name)

        if len(self.known_face_names) == 2:
            self.known_face_encodings.pop(0)
            self.known_face_names.pop(0)

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)

        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            # use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            #
            best_match_index = int(np.argmin(face_distances))
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]

            face_names.append(name)

        # Convert to numpy array to adjust coordinates with frame resizing quickly
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names

    def convertBinaryToFile(binarydata, filename):
        with open(filename, 'wb') as file:
            file.write(binarydata)
