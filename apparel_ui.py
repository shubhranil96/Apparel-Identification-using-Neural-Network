import pandas as pd
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from PIL import Image
import Training
import Testing
from tensorflow.keras.models import load_model
from keras import backend as K
import h5py
import streamlit as st
showpred = 0

apparel = ["T-shirt / Top","Trouser", "Pullover", "Dress", "Coat" ,"Sandal", "Shirt" , "Sneaker", "Bag", "Ankle Boot"]
st.title("Apparel Identification")
st.header("** Using Neural Network **")
st.write("")
st.write("")

st.write("Pick an image from the left, You'll be able to view the image")

st.write("When you're ready, submit a prediction on the left.")

st.write("")
st.write("")

st.sidebar.title("About")

st.sidebar.info("This is an application used to demonstrate the identification of an apparel from its image. The aim is to take any image and give the results as accurately as possible")

st.sidebar.header("Train Neural Network")

if st.sidebar.button("Train CNN"):
    if os.path.exists('./models/'):
        st.success("Model trained")
    else:
        with st.spinner("Training in progress .."):
            Training.train()
        st.success("Finished")


st.sidebar.header("Predict New Images")

onlyfiles = [f for f in listdir("C:/Users/Shreya/Final project/input folder/") if isfile(join("C:/Users/Shreya/Final project/input folder/", f))]
demo = st.sidebar.selectbox("Pick an image", onlyfiles)
img = Image.open("C:/Users/Shreya/Final project/input folder/"+demo)
st.image(img, width=300)

if st.sidebar.button("Predict Apparel"):
    if os.path.exists('./models/'):
        model_path = './models/model.h5'
        model_weights_path = './models/weights.h5'
        model = load_model(model_path)
        model.load_weights(model_weights_path)
        showpred = 1
        prediction = Testing.predict(model, "C:/Users/Shreya/Final project/input folder/"+demo)
    else: 
        st.warning("Need to train model")
    
    if showpred == 1:
        st.write("** This is a **", apparel[prediction[0]])
    
