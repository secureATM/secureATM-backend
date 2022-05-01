# import base64
# import io
# import json
# import os
# from PIL.Image import Image
from flask import Flask, Response, send_file, request, jsonify
from flask_cors import CORS, cross_origin
import cv2
import numpy as np
from simple_facerec import SimpleFacerec
from pymongo import MongoClient
from datetime import datetime
from facial_expression import FacialExpression

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Suprised"}

load_facial_expression_model = FacialExpression()

Facerec = SimpleFacerec()
Facerec.load_encoding_images("Unknown")

app = Flask(__name__)
CORS(app)

user_name = ""
age = ""
phone_number = ""

detection_1 = ""
detection_2 = ""
detection_3 = ""
emotion_name = ""
count = 0
account_number = ""
current_account_number = ""
pics = ""

gun_detect_count = 0
angry_count = 0
disgusted_count = 0
fearful_count = 0
happy_count = 0
neutral_count = 0
sad_count = 0
suprised_count = 0
emotion_count = []
acc_num_count = 0
detect_face_count = 0
face_rec_count = 1

facial_rec_result = ""
weapon_detect_result = ""
facial_expression_result = ""
display_model_headline = ""

# pic_folder = os.path.join('Images')
# app.config['UPLOAD_FOLDER'] = pic_folder

# default_pic_folder = os.path.join('DefaultImages')
# app.config['DEFAULT_UPLOAD_FOLDER'] = default_pic_folder

client = MongoClient(
    'mongodb+srv://kavishka:kav1234@cluster0.ddfkm.mongodb.net/myFirstDatabase?retryWrites=true&w=majority')
db = client.get_database('test-db')
records = db.atm_user
newCol = db.detection_details


@app.route('/detectionOutput', methods=['POST'])
def detectionOutput():
    global facial_rec_result, weapon_detect_result, facial_expression_result, count, detection_1, detection_2, \
        detection_3, account_number, gun_detect_count, angry_count, disgusted_count, fearful_count, happy_count, \
        neutral_count, sad_count, suprised_count, emotion_count, user_name, age, phone_number

    facial_rec_result = request.form["facialRecognition"]
    weapon_detect_result = request.form["weaponDetection"]
    facial_expression_result = request.form["facialExpression"]
    account_num = request.form["accountNum"]
    phone_num = request.form["phoneNum"]
    stop = request.form["stop"]
    now = datetime.now()
    date = now.strftime("%d/%m/%Y %H:%M:%S")

    if stop == "true":
        count = 0
        gun_detect_count = 0
        angry_count = 0
        disgusted_count = 0
        fearful_count = 0
        happy_count = 0
        neutral_count = 0
        sad_count = 0
        suprised_count = 0
        emotion_count = []
        user_name = ""
        age = ""
        phone_number = ""

    newDict = {"date": date, "account": account_num, "phoneNum": phone_num, "detect_1": facial_rec_result,
               "detect_2": weapon_detect_result,
               "detect_3": facial_expression_result}
    if account_number != "":
        newCol.insert_one(newDict)

    detection_1 = ""
    detection_2 = ""
    detection_3 = ""
    account_number = ""
    return ""


@app.route('/getUserHistory', methods=['GET'])
def getUserHistory():
    data = []
    for doc in newCol.find({}):
        doc['_id'] = str(doc['_id'])
        data.append(doc)
    return jsonify(data)


@app.route('/accNum', methods=['POST'])
def accNum():
    global account_number, user_name, current_account_number, acc_num_count
    account_number = request.form["accNum"]
    if account_number != "":
        acc_num_count += 1
    print(account_number)
    return ""


@app.route('/img', methods=['GET'])
def atm_user_dp():
    if account_number != "":
        global pics
        ob = records.find_one({"accNum": account_number})
        pics = ob["dp"]
        return send_file(pics, mimetype='image/jpg')
    else:
        # print("false")
        # pics = os.path.join(app.config['DEFAULT_UPLOAD_FOLDER'], 'user.jpg')
        # print(pics)
        # return send_file(pics, mimetype='image/jpg')
        return ""


@app.route('/atm_user_details', methods=['GET'])
def user_details():
    global count, user_name, face_rec_count, age, phone_number
    if account_number != "":
        ob = records.find_one({"accNum": account_number})
        user_name = ob["name"]
        age = ob["age"]
        phone_number = ob["phone_num"]

        if acc_num_count > 0:
            face_rec_count = 0
            Facerec.load_encoding_images(ob["name"])

        return {"accNum": account_number, "name": user_name, "age": age, "phone_num": phone_number}
    else:
        return {"accNum": "", "name": "", "age": "", "phone_num": ""}


@app.route('/live', methods=['GET'])
@cross_origin()
def test():
    return Response(video(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route('/live/output', methods=['GET'])
def output():
    global detection_1, detection_2, detection_3, display_model_headline
    return {"detection_1": detection_1, "detection_2": detection_2, "detection_3": detection_3,
            "model_headline": display_model_headline}


def video():
    global display_model_headline
    watch_cascade = cv2.CascadeClassifier('cascade_files/gun_cascade.xml')
    face_detector = cv2.CascadeClassifier('cascade_files/haarcascade_frontalface_default.xml')

    vid = cv2.VideoCapture(0)

    while True:
        ret, frame = vid.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        num_faces = face_detector.detectMultiScale(gray, 1.3, 5)
        rectangles = watch_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=10, minSize=(75, 75))

        global count
        if 200 > count > 0:
            if len(rectangles) > 0:
                global detection_2, gun_detect_count
                if gun_detect_count > 120:
                    detection_2 = "weapon detect"
                else:
                    detection_2 = "not detect"

                for (i, (x, y, w, h)) in enumerate(rectangles):
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, "Gun" + " #{}".format(i + 1), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (0, 0, 255), 2)

                gun_detect_count += 1
        print(count)

        if 500 > count > 200:
            global detection_3, angry_count, disgusted_count, fearful_count, happy_count, neutral_count, sad_count, suprised_count
            for (x, y, w, h) in num_faces:
                cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 4)
                roi_gray_frame = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

                emotion_prediction = load_facial_expression_model.detect_facial_emotions(cropped_img)
                maxindex = int(np.argmax(emotion_prediction))

                cv2.putText(frame, emotion_dict[maxindex], (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                            cv2.LINE_AA)
                if maxindex == 0:
                    angry_count += 1
                elif maxindex == 1:
                    disgusted_count += 1
                elif maxindex == 2:
                    fearful_count += 1
                elif maxindex == 3:
                    happy_count += 1
                elif maxindex == 4:
                    neutral_count += 1
                elif maxindex == 5:
                    sad_count += 1
                elif maxindex == 6:
                    suprised_count += 1

                if count > 400:
                    emotion_count.insert(0, angry_count)
                    emotion_count.insert(1, disgusted_count)
                    emotion_count.insert(2, fearful_count)
                    emotion_count.insert(3, happy_count)
                    emotion_count.insert(4, neutral_count)
                    emotion_count.insert(5, sad_count)
                    emotion_count.insert(6, suprised_count)
                    detection_3 = emotion_dict[emotion_count.index(max(emotion_count))]

        global detection_1, detect_face_count, face_rec_count, user_name
        if count > 510:
            if face_rec_count > 60:
                face_locations, face_names = Facerec.detect_known_faces(frame)

                for face_loc, face_name in zip(face_locations, face_names):
                    y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

                    cv2.putText(frame, face_name, (x1, int(y1 - 10)), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

                    if user_name == face_name:
                        detect_face_count += 1
                    if detect_face_count > 20:
                        detection_1 = user_name
                    elif detect_face_count < 20:
                        detection_1 = "Unknown"

            face_rec_count += 1

        if count < 200:
            display_model_headline = "Detecting weapons..."
        elif count < 500:
            display_model_headline = "Analyzing facial expression.."
        elif count > 510:
            display_model_headline = "Facial recognition..."

        (flag, encodedImage) = cv2.imencode('.jpg', frame)
        yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n'
        cv2.waitKey(40)
        count = count + 1


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8000, debug=True)
