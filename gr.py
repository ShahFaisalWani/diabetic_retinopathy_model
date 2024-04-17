import gradio as gr
import tensorflow as tf
import cv2
import numpy as np

def predict_input_image(image):
    resized_image = cv2.resize(image, (224, 224))
    resized_image = resized_image / 255.0
    new_model = tf.keras.models.load_model("64x3-CNN.keras")
    predict = new_model.predict(np.array([resized_image]))
    formatted_output = {"DR": predict[0][0], "NO_DR": predict[0][1]}
    per = np.argmax(predict, axis=1)
    if per == 1:       
        print('Diabetic Retinopathy Not Detected') 
    else:
        print('Diabetic Retinopathy Detected')
    return formatted_output    

def image_classifier(inp):    
    prediction = predict_input_image(inp)
    return prediction

demo = gr.Interface(fn=image_classifier, inputs="image", outputs="label")
demo.launch()
