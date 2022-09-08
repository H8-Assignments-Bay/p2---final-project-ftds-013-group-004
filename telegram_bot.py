from cgitb import text
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from io import BytesIO
import io
from PIL import Image
import cv2
from keras.applications.resnet import ResNet50,preprocess_input 
#from tensorflow.python.keras.applications.resnet50 import ResNet50,preprocess_input


import numpy as np
import pickle
import tensorflow
from keras.preprocessing import image
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import requests
from io import BytesIO



def start(update, context):
    update.message.reply_text("Halo, Selamat Datang di Belanja Boss E-commerce")
    update.message.reply_text("Feel free to ask question")

def info(update, context):
    update.message.reply_text("Ini adalah Bot Belanja Boss")
    update.message.reply_text("Kamu mau belanja apa? ini link toko kami https://belanja-boss-e-commerce.anvil.app/")

def main(update, context):
    update.message.reply_text("Saya Memikirkan Apa, Coba Tebak")

def echo(update, context):
    update.message.reply_text(update.message.text)    


    
# Load file dan model

# url directory berisikan link pada setiap product fashion
url_directory = pickle.load(open('url_directory.pkl','rb'))

# untuk melakukan feature extraction pada gambar
model = tensorflow.keras.models.load_model("recommender.h5")

# recommendation model untuk mendapatkan rekomendasi fashion
recommendation_model = pickle.load(open('model_similarity.pkl','rb'))

def load_image(img):
    photo_file = updater.getFile(updater.message.photo[-1].file_id)
    filename = '{}.jpg'.format(photo_file.file_id)
    photo_file.download(filename)
    with io.open(filename, 'rb') as image_file:
        content = image_file.read()
    uploaded_files = Image.open(img.read()).resize((224,224))
    uploaded_files = np.array(uploaded_files)
    data           = uploaded_files[np.newaxis,...]
    return data

def feature_extraction(img,model):
    preprocessed_img = preprocess_input(img)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def recommend(features):#feature_list):
    distances, indices = recommendation_model.kneighbors([features])
    return indices

def imager(updater, context):
    image_user = updater.message.photo[-1].get_file()
    image_user.download("img.jpg")
    img = cv2.imread("img.jpg")
    img = cv2.resize(img, (224,224))
    img = np.reshape(img, (1,224,224,3))
    features = feature_extraction(img,model)
    indices = recommend(features)
    updater.message.reply_text("Rekomendasi 1")
    updater.message.reply_text(url_directory[indices[0][0]])
    updater.message.reply_text("Rekomendasi 2")
    updater.message.reply_text(url_directory[indices[0][1]])
    updater.message.reply_text("Rekomendasi 3")
    updater.message.reply_text(url_directory[indices[0][2]])
    updater.message.reply_text("Rekomendasi 4")
    updater.message.reply_text(url_directory[indices[0][3]])
    updater.message.reply_text("Rekomendasi 5")
    updater.message.reply_text(url_directory[indices[0][4]])

# MAIN PROGRAM
TOKEN = '5750120669:AAHmmXgf4OwgJBznEHIXp6ODqfC6Yan1hUE'
updater = Updater(TOKEN, use_context=True)

dp = updater.dispatcher

# Menambah Command
dp.add_handler(CommandHandler("start", start))
dp.add_handler(CommandHandler("info", info))
dp.add_handler(CommandHandler("main", main))

# Menambah Message Handler
##dp.add_handler(MessageHandler(Filters.photo, image))
dp.add_handler(MessageHandler(Filters.photo, imager))
##dp.add_handler(MessageHandler(Filters.photo, save))
dp.add_handler(MessageHandler(Filters.text, echo))

# run bot
updater.start_polling()
updater.idle()
