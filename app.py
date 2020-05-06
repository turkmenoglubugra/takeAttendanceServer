from flask import Flask, render_template,request,redirect,url_for,jsonify
from flask import Response
from datetime import datetime
from flask_json import FlaskJSON, JsonError, json_response, as_json
import json
import base64
import sys
import os
import numpy as np
import imutils
import pickle
import cv2
import os
import pymysql
import datetime
from datetime import datetime

app = Flask(__name__)

@app.route('/')
def index():
    return "asdasdasd!"


@app.route('/get_courses', methods=['GET'])
def api_hello():
    db = pymysql.connect("localhost", "root", "17489625Bkt", "check_up")

    cursor = db.cursor()

    cursor.execute("SELECT * FROM course")

    myresult = cursor.fetchall()

    my_list = []

    for x in myresult:
        my_list.append(x[1])

    data = json.dumps(my_list)
    db.close()

    resp = Response(data, status=200, mimetype='application/json')

    return resp

@app.route('/user', methods=['POST'])
def asdasd():
    person_dict = json.loads(request.data)

    db = pymysql.connect("localhost", "root", "17489625Bkt", "check_up")

    cursor = db.cursor()
    print(person_dict['key2'])
    sql = "INSERT INTO polling (course_id) VALUES (%s)"
    val = person_dict['key2']
    cursor.execute(sql, val)

    sql = "SELECT LAST_INSERT_ID()"

    cursor.execute(sql)

    myresult = cursor.fetchall()
    val = myresult[0]
    val1 = myresult[0]
    count = 1

    #print(person_dict['key1'])
    with open("imageToSave.png", "wb") as fh:
        fh.write(base64.decodebytes(bytes(person_dict['key1'], encoding='utf-8')))
    #   try:
    #yüz tespiti yapan modelin diskten yuklenmesi
    #sql = "INSERT INTO images (pollingID, image) VALUES (%s, %s)"
    #val2 = person_dict['key1']
    #val3 = (val1, val2)
    #cursor.execute(sql, val3)

    print("[INFO] loading face detector...")
    protoPath = os.path.sep.join(["face_detection_model", "deploy.prototxt"])
    modelPath = os.path.sep.join(["face_detection_model",
                                  "res10_300x300_ssd_iter_140000.caffemodel"])
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # ozellik cikarimi yapan modelin diskten yuklenmesi
    print("[INFO] loading face recognizer...")
    embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

    # makine ögrenmesi modelinin ve label encoderin diskten yuklenmesi
    recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read())
    le = pickle.loads(open("output/le.pickle", "rb").read())

    # Resimleri yükledikten sonra genişliğinim 600 yapılması
    # Resimlerin boyutlarının elde edilmesi
    image = cv2.imread("imageToSave.png")
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    # blob insa etme
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # BLob'un yüz bulma modeline input olarak verilmesi
    # Görüntüdeki yüzlerin elde edilmesi
    detector.setInput(imageBlob)
    detections = detector.forward()
    names = []
    # tespitler
    count_unknown = 0
    for i in range(0, detections.shape[2]):
        # confidence degeri cikarma
        confidence = detections[0, 0, i, 2]

        # zayif tespitleri eleme
        if confidence > 0.5:
            # tespit edilen yuzun cercevesini hesaplama
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # yuzun boyutlarini cikarma
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # yuzun buyuklugunden emin olma
            if fW < 20 or fH < 20:
                continue

            # tespit edilen yuzden blob insa etme
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
                                             (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # siniflandirma yapma
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            if (proba * 100 > 70):
                name = le.classes_[j]
                names.append(name)
                count = count + 1

            # yuze ait oldugu sinifi ve basari oranini gosterme
            if (proba * 100 > 70):
                asd = name
                print(proba * 100)
            else:
                count_unknown = count_unknown + 1
                asd = "unknown"
            print(asd)
            text = "{}: {:.2f}%".format(asd, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)
            cv2.putText(image, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    data = pickle.loads(open("output/embeddings.pickle", "rb").read())

    # resimdekileri saydirma
    index = 0
    for i in range(len(data["names"])):
        if (data["names"][i] != data["names"][i - 1]):
            index = index + 1
            control = "-"
            for j in range(len(names)):
                if (names[j] == data["names"][i]):
                    control = "+"

            if (control == "+"):
                name1 = data["names"][i]
                sql = "INSERT INTO students (name, polling_id, state) VALUES (%s,%s,%s)"
                cursor.execute(sql,(name1,val, '+'))
                db.commit()
            if (control == "-"):
                name2 = data["names"][i]
                print("buraya girdi")
                sql = "INSERT INTO students (name, polling_id, state) VALUES (%s,%s,%s)"
                cursor.execute(sql, (name2, val,'-'))
                db.commit()

    print("Sınıftaki tanınmayan öğrenci sayısı: {}".format(count_unknown))
    if (len(names) > 0):
        print("Sınıftaki tanınan öğrenci sayısı: {}".format(len(names)))

    # cikis resmini gosterme
    image = imutils.resize(image, width=500, height=300)
    cv2.imwrite("images/class_image.jpg", image)
    print("Yoklama Alma Tamamlandı!")

    db.close()

    my_list = []

    d = {"persons":[{"city": "Seattle", "name": "Brian"},{"city": "Amsterdam", "name": "David"}]}
    data = json.dumps(d)
    return Response(data, status=200, mimetype='application/json')

@app.route('/get_dates', methods=['POST'])
def asdasds():
    person_dict = json.loads(request.data)

    db = pymysql.connect("localhost", "root", "17489625Bkt", "check_up")

    cursor = db.cursor()

    sql = "Select * from polling where course_id LIKE %s"
    val = person_dict['key1']
    print(val)
    cursor.execute(sql, val)

    myresult = cursor.fetchall()

    my_list = []

    for x in myresult:
        a = x[2].strftime("%d-%b-%Y (%H:%M:%S)")
        print("girdi")
        print(a)
        my_list.append(a)

    def myconverter(o):
        if isinstance(o, datetime.datetime):
            return o.__str__()

    db.close()
    #print(data)
    data = json.dumps({'results': my_list})

    d = {"persons": [{"city": "Seattle", "name": "Brian"}, {"city": "Amsterdam", "name": "David"}]}
    #data = json.dumps(d)
    return Response(data, status=200, mimetype='application/json')

@app.route('/yoklama', methods=['POST'])
def dsa():
    person_dict = json.loads(request.data)
    print(person_dict['key1'])

    db = pymysql.connect("localhost", "root", "17489625Bkt", "check_up")

    cursor = db.cursor()

    sql = "INSERT INTO image (image) VALUES (%s)"
    val = person_dict['key1']
    cursor.execute(sql, val)

     #a = "sad"
    #cursor.execute("INSERT INTO image (image) VALUES (%s),a")

    db.commit()

    db.close()

    my_list = []

    d = {"persons":[{"city": "Seattle", "name": "Brian"},{"city": "Amsterdam", "name": "David"}]}
    data = json.dumps(d)
    return Response(data, status=200, mimetype='application/json')

@app.route('/get_polling', methods=['POST'])
def asdasdss():
    person_dict = json.loads(request.data)

    db = pymysql.connect("localhost", "root", "17489625Bkt", "check_up")

    cursor = db.cursor()
    sql = "Select id from polling where DateTime LIKE %s"
    val = person_dict['key1']
    datetime_object = datetime.strptime(val, '%d-%b-%Y (%H:%M:%S)')

    print(datetime_object)
    cursor.execute(sql, datetime_object)

    myresult = cursor.fetchall()
    print(myresult[0])

    sql = "Select name, state from students where polling_id LIKE %s"
    val = myresult[0]
    cursor.execute(sql, val)
    myresult = cursor.fetchall()
    print(myresult)


    db.close()
    data = json.dumps({'results': myresult})
    print(data)

    d = {"persons": [{"city": "Seattle", "name": "Brian"}, {"city": "Amsterdam", "name": "David"}]}
    #data = json.dumps(d)
    return Response(data, status=200, mimetype='application/json')

@app.route('/insert_delete_course', methods=['POST'])
def asdasdssss():
    person_dict = json.loads(request.data)

    val = person_dict['key1']

    db = pymysql.connect("localhost", "root", "17489625Bkt", "check_up")

    cursor = db.cursor()

    sql = "INSERT INTO course (course) VALUES (%s)"

    print(val)

    cursor.execute(sql,val)

    db.commit()

    myresult = cursor.fetchall()

    db.close()

    data = json.dumps({'results': val})

    d = {"persons": [{"city": "Seattle", "name": "Brian"}, {"city": "Amsterdam", "name": "David"}]}

    return Response(data, status=200, mimetype='application/json')


@app.route('/get_coursesid', methods=['GET'])
def api_helloa():
    db = pymysql.connect("localhost", "root", "17489625Bkt", "check_up")

    cursor = db.cursor()

    cursor.execute("SELECT * FROM course")

    myresult = cursor.fetchall()

    my_list = []

    for x in myresult:
        my_list.append(x[0])

    data = json.dumps(my_list)
    db.close()

    resp = Response(data, status=200, mimetype='application/json')

    return resp

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
