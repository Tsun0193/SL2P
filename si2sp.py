import cv2
import mediapipe as mp
import numpy as np
import json
import tensorflow as tf
import pandas as pd
import os
import time
import warnings
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Form
from src.assets.models.llm.llm import LLM

from pydantic import BaseModel
from typing import Annotated
from fastapi.middleware.cors import CORSMiddleware

class Item(BaseModel):
    name: str

warnings.filterwarnings('ignore')

app = FastAPI()

origins = [
    "http://localhost:4200",
    "http://127.0.0.1:4200",
    "http://127.0.0.1:4200/",
    "http://localhost:4200/"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
    
)

load_dotenv()
completion = LLM()

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils 
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image.flags.writeable = False
    pred = model.process(image) 
    image.flags.writeable = True  
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    return image, pred


def draw(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                              mp_drawing.DrawingSpec(color=(0,0,255), thickness=3, circle_radius=3),
                              mp_drawing.DrawingSpec(color=(0,0,0), thickness=1, circle_radius=0))
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(0,150,0), thickness=3, circle_radius=3),
                              mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(200,56,12), thickness=3, circle_radius=3),
                              mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(250,56,12), thickness=3, circle_radius=3),
                              mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2))

def extract_coordinates(results):
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]) if results.face_landmarks else np.zeros((468, 3))
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33, 3))
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 3))
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 3))
    res = np.concatenate([face, pose, lh, rh])
    return res

class config:
    path = os.getcwd() + '/src/assets/models/si2sp/'
    seq_len = 12
    rpf = 543
    model_path = path + 'results/asl_model/model.tflite'

def load_relevant(path):
    data_cols = ['x', 'y', 'z']
    data = pd.read_parquet(path, columns=data_cols)
    n_frames = int(len(data) / config.rpf)
    data = data.values.reshape(n_frames, config.rpf, 3) # len(data_cols) = 3
    return data.astype(np.float32)

def load_json(_path):
    with open(config.path + _path, 'r') as f:
        return json.load(f)
    

sign_map = load_json('sign_to_prediction_index_map.json')
s2p_map = {
    k.lower(): v for k, v in sign_map.items()
}
p2s_map = {
    v: k for k, v in sign_map.items()
}
encoder = lambda x: s2p_map.get(x.lower())
decoder = lambda x: p2s_map.get(x)

model = tf.lite.Interpreter(model_path=config.model_path)


found_signs = list(model.get_signature_list().keys())
prediction_fn = model.get_signature_runner('serving_default')

@app.post("/test")
async def create_item(uploadFile: Annotated[UploadFile, Form()]):
    print(uploadFile)
    return {"name": uploadFile.filename}

@app.get('/live-translator')
async def live_translate():
    seq = []
    cap = cv2.VideoCapture(0)
    preds = ['']
    curr_len = len(preds)
    start_time = time.time()
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            img, results = mediapipe_detection(frame, holistic)
            draw(img, results)

            landmarks = extract_coordinates(results)
            seq.append(landmarks)
            if len(seq) % 15 == 0:
                prediction = prediction_fn(inputs=np.array(seq, dtype = np.float32))
                sign = np.argmax(prediction["outputs"])
                sign = decoder(sign)
                if preds[-1] != sign:
                    preds.append(sign)
                
            cv2.imshow('Sign Language Detection', img)

            # every 3 seconds, print preds to the terminal, get time from browser
            if len(preds) > curr_len and time.time() - start_time > 3 and len(preds) > 1:
                text = ' '.join(preds)
                print(preds)
                print(completion(text))
                start_time = time.time()
                curr_len = len(preds)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

# get input as a video file
@app.post('/predict')
async def predict(file: UploadFile = File(...), path = 'data'):
    with open(f"{path}/{file.filename}", "wb") as buffer:
        buffer.write(file.file.read())

    cap = cv2.VideoCapture(f"{path}/{file.filename}")
    os.remove(f"{path}/{file.filename}")
    seq = []
    preds = ['']
    curr_len = len(preds)
    start_time = time.time()

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            img, results = mediapipe_detection(frame, holistic)
            draw(img, results)

            landmarks = extract_coordinates(results)
            seq.append(landmarks)
            if len(seq) % 15 == 0:
                prediction = prediction_fn(inputs=np.array(seq, dtype = np.float32))
                sign = np.argmax(prediction["outputs"])
                sign = decoder(sign)
                if preds[-1] != sign:
                    preds.append(sign)
                
            # cv2.imshow('Sign Language Detection', img)

            # every 3 seconds, print preds to the terminal, get time from browser
            if len(preds) > curr_len and time.time() - start_time > 3:
                print(' '.join(preds))
                start_time = time.time()
                curr_len = len(preds)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    live_translate()
    # predict('path/to/video')



