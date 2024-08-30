import face_recognition

class FaceRecognitionAgent:
    def __init__(self):
        self.known_face_encodings = []  # Load known face encodings here
        self.known_face_names = []      # Load known face names here

    def load_known_faces(self, image_paths, names):
        for image_path, name in zip(image_paths, names):
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)[0]
            self.known_face_encodings.append(encoding)
            self.known_face_names.append(name)

    def recognize_face(self, image_path):
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            if True in matches:
                match_index = matches.index(True)
                return self.known_face_names[match_index]
        return "Unknown"
