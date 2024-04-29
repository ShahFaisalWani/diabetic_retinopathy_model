from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
import cv2
import numpy as np

app = FastAPI()

model = tf.keras.models.load_model("64x3-CNN.keras")

Retina_classes = ['DR', 'No_DR']

def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    return img

def predict_input_image(img):
    img = preprocess_image(img)
    img = img.reshape(-1, 224, 224, 3)
    predictions = model.predict(img)[0]
    result = {Retina_classes[i]: float(predictions[i]) for i in range(2)}
    return result

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    prediction = predict_input_image(img)
    return JSONResponse(content=prediction)
