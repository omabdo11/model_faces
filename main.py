from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import face_recognition
import os
import json
from datetime import datetime
from typing import List

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# إعدادات المجلد
IMAGES_PATH = "attendanceImages"
ATTENDANCE_FILE = "attendance.json"
RTSP_URL = 0  # يمكن تغييره إلى رابط كاميرا RTSP

# تحميل الصور
def load_images_from_folder(path):
    images = []
    class_names = []
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path, filename))
        if img is not None:
            images.append(img)
            class_names.append(os.path.splitext(filename)[0])
    return images, class_names

# ترميز الصور
def find_encodings(images):
    encodings = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        if encode:
            encodings.append(encode[0])
    return encodings

images, classNames = load_images_from_folder(IMAGES_PATH)
encodeListKnown = find_encodings(images)

# حفظ الحضور
def mark_attendance(name: str, filename=ATTENDANCE_FILE):
    entry = {
        "name": name,
        "time": datetime.now().strftime('%H:%M:%S')
    }
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except:
        data = []

    if not any(d['name'] == name for d in data):
        data.append(entry)
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Attendance marked for {name}")

# API: تسجيل الحضور بصورة
@app.post("/api/mark_attendance")
async def mark_attendance_api(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        small_img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        rgb_small_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)

        faces_loc = face_recognition.face_locations(rgb_small_img)
        faces_enc = face_recognition.face_encodings(rgb_small_img, faces_loc)

        recognized_names = []
        for face_encoding in faces_enc:
            matches = face_recognition.compare_faces(encodeListKnown, face_encoding)
            face_dis = face_recognition.face_distance(encodeListKnown, face_encoding)
            best_match = np.argmin(face_dis)

            if matches[best_match]:
                name = classNames[best_match].upper()
                recognized_names.append(name)
                mark_attendance(name)

        return {"recognized": recognized_names}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# API: مسح بيانات الحضور
@app.post("/api/clear_attendance")
def clear_attendance_api():
    try:
        with open(ATTENDANCE_FILE, 'w') as f:
            json.dump([], f, indent=4)
        return {"message": "Attendance data cleared successfully!"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# API: جلب بيانات الحضور
@app.get("/api/get_attendance")
def get_attendance_api():
    try:
        with open(ATTENDANCE_FILE, 'r') as f:
            data = json.load(f)
        return {"attendance": data}
    except:
        return {"attendance": []}

# API: بث الكاميرا
def gen_frames():
    cap = cv2.VideoCapture(RTSP_URL)
    frame_resizing = 0.25

    while True:
        success, img = cap.read()
        if not success:
            break

        small_img = cv2.resize(img, (0, 0), fx=frame_resizing, fy=frame_resizing)
        rgb_small_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)

        faces_loc = face_recognition.face_locations(rgb_small_img)
        faces_enc = face_recognition.face_encodings(rgb_small_img, faces_loc)

        for encodeFace, faceLoc in zip(faces_enc, faces_loc):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                y1, x2, y2, x1 = [int(i / frame_resizing) for i in faceLoc]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                mark_attendance(name)

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")
