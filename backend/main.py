from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from io import BytesIO
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
from PIL import Image
import uvicorn

def load_final_model():
    model = load_model('./model/model_weights.h5')
    print("Model loaded")
    return model


def predict(img: Image.Image):
    img = img.resize((64, 64))
    img = image.img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, axis = 0)

    classifier = load_final_model()

    prediction = classifier.predict(img, batch_size = None, steps = 1)[0] * 100
    prediction = prediction.tolist()
    return prediction

def read_imagefile(file) -> Image.Image:
    img = Image.open(BytesIO(file)) #Pillow object
    return img


app = FastAPI()
 
@app.get('/test')
async def testing():
    return "Hello World"

@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):

    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return {"Error" : "Image must be jpg or png format!"}
        
    image = read_imagefile(await file.read())

    prediction = predict(image)

    final_prediction = dict()

    final_prediction.update({"Dog Percentage" : prediction[0]})
    final_prediction.update({"Cat Percentage" : 100 - prediction[0]})
    if prediction[0] >= 50:
        final_prediction.update({"Label" : "Dog"})
    
    elif prediction[0] < 50:
        final_prediction.update({"Label" : "Cat"})

    return final_prediction

if __name__ == "__main__":
    uvicorn.run(app, debug = True)