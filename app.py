import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import requests
from io import BytesIO
st.set_page_config(layout="centered", page_title="Fashion Recommender")

# Load directory url, feature extraction model, recommendation model
url_directory = pickle.load(open('url_directory.pkl','rb')) # berisikan directory url product
model = tensorflow.keras.models.load_model("recommender.h5") # feature extraction model
recommendation_model = pickle.load(open('model_similarity.pkl','rb')) # recommendation model

# Initialize Image = None atau tidak ada
img = None

# Home Page
st.markdown("<h1 style='text-align: center; color: white;'>Fashion Recommender System</h1>", unsafe_allow_html=True)
gambar_homepage = Image.open('homepage.jpg')
st.image(gambar_homepage, use_column_width = True, caption='Final Project Hacktiv8 Batch 013 Group 04')

# About
st.subheader("About")
st.write("""This project is about Fashion where we try to give an experience to customers by giving recommendations based on their preferences in selecting their outfits.""")

def load_image(img):
    img = tensorflow.keras.utils.load_img(img, target_size=(224,224,3)) 
    x = tensorflow.keras.utils.img_to_array(img) 
    x = np.expand_dims(x, axis=0)
    return x

def feature_extraction(img,model):
    preprocessed_img = preprocess_input(img)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def recommend(features):
    distances, indices = recommendation_model.kneighbors([features])
    return indices

st.subheader("Upload Clothing Image Preferences: ")
uploaded_file = st.file_uploader("Choose an image",type=["jpg", "png", "jpeg"])

st.subheader("Your Image: ")
if uploaded_file is not None:
    gambar = Image.open(uploaded_file)
    st.image(gambar)

st.subheader(" Our recommendation based on our catalog : ")
if uploaded_file is not None:
    button_recommend = st.button('Give Recommendation')
    if button_recommend:
        # display the file
        image = load_image(uploaded_file)
        # feature extract
        features = feature_extraction(image,model)
        #recommendation
        indices = recommend(features)
        # show
        col1,col2,col3,col4,col5 = st.columns(5)
        with col1:
            response_1 = requests.get(url_directory[indices[0][0]])
            img_1 = Image.open(BytesIO(response_1.content))
            st.image(img_1)
        with col2:
            response_2 = requests.get(url_directory[indices[0][1]])
            img_2 = Image.open(BytesIO(response_2.content))
            st.image(img_2)
        with col3:
            response_3 = requests.get(url_directory[indices[0][2]])
            img_3 = Image.open(BytesIO(response_3.content))
            st.image(img_3)
        with col4:
            response_4 = requests.get(url_directory[indices[0][3]])
            img_4 = Image.open(BytesIO(response_4.content))
            st.image(img_4)
        with col5:
            response_5 = requests.get(url_directory[indices[0][4]])
            img_5 = Image.open(BytesIO(response_5.content))
            st.image(img_5)
