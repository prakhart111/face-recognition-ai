from fastapi import FastAPI, File, UploadFile
from tensorflow.python.keras.models import load_model
from io import BytesIO
# import numpy as np
# import tensorflow as tf
from PIL import Image
import uvicorn

def load_final_model():
    model = load_model('./model/model_weights.h5')
    print("Model loaded")
    return model

def img_preprocess(img):
    # resize to 128x128
    img = img.resize((128, 128))
    return img

def predictSimilarImage(anchor: Image.Image, img1: Image.Image, img2: Image.Image):
    anchor = img_preprocess(anchor)
    img1 = img_preprocess(img1)
    img2 = img_preprocess(img2)

    classifier = load_final_model()

    # jiska zero aaya that's +ve 
    anchorWithImg1Test = classifier.classify_images(anchor, img1) # 0 
    # jiska one aaya that's -ve
    anchorWithImg2Test = classifier.classify_images(anchor, img2) # 1


    if anchorWithImg1Test <= anchorWithImg2Test:
        # return "Image 1 is more similar to Anchor Image"
        return True
    else:
        # return "Image 2 is more similar to Anchor Image"
        return False
 
 # for api
def read_imagefile(file) -> Image.Image:
    img = Image.open(BytesIO(file)) #Pillow object
    return img

def file_check(file):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return False
    return True


app = FastAPI()
 
@app.get('/test')
async def testing():
    return "Hello World"

@app.post("/predict")
async def predict_api(fileAnchor: UploadFile = File(...), file1: UploadFile = File(...), file2: UploadFile = File(...)):

    # extension check
    if not file_check(fileAnchor):
        return {"Error" : "Image must be jpg or png format!"}
    if not file_check(file1):
        return {"Error" : "Image must be jpg or png format!"}
    if not file_check(file2):
        return {"Error" : "Image must be jpg or png format!"}
        
    anchor = read_imagefile(await fileAnchor.read())
    img1 = read_imagefile(await file1.read())
    img2 = read_imagefile(await file2.read())

    try:
        prediction = predictSimilarImage(anchor, img1, img2)
    except Exception as e:
        return {"Error": str(e)}

    final_prediction = dict()

    if prediction:
        final_prediction.update({"Result" : "Image 1 is more similar to Anchor Image"})
    else:
        final_prediction.update({"Label" : "Image 2 is more similar to Anchor Image"})

    return final_prediction

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)