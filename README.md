# Face Similarity Detection AI

- B.Tech. Final year project on Face Similarity Detection AI using Siamese Networks.
- The project is built using TensorFlow and Keras.
- The model is trained on the **Extracted Faces** dataset from the **face-recognition-dataset**.
- The model uses **Triplet Loss** to minimize the distance between the anchor and positive images and maximize the distance between the anchor and negative images.
- Model is exported as "H5" and wrapped in a Fast API.
- The API is to be deployed on AWS as a containerized service, using Docker.
- A React Frontend is built to interact with the API.

## Project Structure

- **backend**: Contains the model and API. ( Tech Stack: FastAPI, TensorFlow, Keras, Pillow, AWS )
- **frontend**: Contains the React Frontend. ( Tech Stack: React, Axios, TailwindCSS )

## Project Setup

### Backend

- Install the required packages using `pip install -r requirements.txt`.
- Run the API using `uvicorn main:app --reload`.

- Working

- Sample 1: 
<img width="810" alt="Sample_1" src="https://github.com/prakhart111/face-recognition-ai/assets/86708181/3394bfe6-043c-4566-8854-357395b5a1bf">

- Sample 2: 
<img width="810" alt="Sample_2" src="https://github.com/prakhart111/face-recognition-ai/assets/86708181/319899be-ab60-49b3-8816-a5316b82c660">

### Frontend

- Install the required packages using `npm install`.
- Run the frontend using `npm start`.

## What it does

- The model takes 3 input images, (anchor, image-1, image-2) and gives a distance value that indicates which of these 2 images are similar or dissimilar to the anchor image.
- Sample UI for the project:

<img src="./frontend/ui.jpg" alt="UI Design" width="70%" />

## Team

Prakhar Tandon, Vaibhav Shankhwar, Abhishek Agrahri and Mansi Purwar. <br/>
Final B.Tech. Computer Science & Engineering. <br/>
Under Guidance of Dr. Vivek Verma, Professor, CSE Dept, HBTU Kanpur.<br/>
