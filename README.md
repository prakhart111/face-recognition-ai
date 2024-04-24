# Face Similarity Detection AI

- B.Tech. Final year project on Face Similarity Detection AI using Siamese Networks.
- The project is built using TensorFlow and Keras.
- The model is trained on the **Extracted Faces** dataset from the **face-recognition-dataset**.
- The model uses **Triplet Loss** to minimize the distance between the anchor and positive images and maximize the distance between the anchor and negative images.
- Model is exported as "H5" and wrapped in a Fast API.
- The API is to be deployed on AWS as a containerized service, using Docker.
- A React Frontend is built to interact with the API.

## Project Structure

- **backend**: Contains the model and API.
- **frontend**: Contains the React Frontend.

## Project Setup

### Backend

- Install the required packages using `pip install -r requirements.txt`.
- Run the API using `uvicorn main:app --reload`.

### Frontend

- Install the required packages using `npm install`.
- Run the frontend using `npm start`.

## What it does

- The model takes 3 input images, (anchor, image-1, image-2) and gives a distance value that indicates which of these 2 images are similar or dissimilar to the anchor image.
- Sample UI for the project:

<img src="./frontend/ui.jpg" alt="UI Design" width="70%" />

## Team

Prakhar Tandon, Vaibhav Shakhvar, Abhishek Agrahri and Mansi Purwar. <br/>
Final B.Tech. Computer Science & Engineering. <br/>
Under Guidance of Dr. Vivek Verma, Assistant Professor, CSE Dept, HBTU.<br/>
