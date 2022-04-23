from keras.models import model_from_json



class FacialExpression:
    def detect_facial_emotions(self, frame):
        json_file = open("model/emotion_model.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        emotion_model = model_from_json(loaded_model_json)

        # load weights into new model
        emotion_model.load_weights("model/emotion_model.h5")
        return emotion_model.predict(frame)
