#!/usr/bin/env python
# coding: utf-8

# In[21]:


#%%writefile emotion_detection.py
import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
from audio_recorder_streamlit import audio_recorder
model=tf.keras.models.load_model(r"emotion detection through audio-2.9.h5")
html_temp = """ 
  <div style="background-color:pink ;padding:10px">
  <h2 style="color:white;text-align:center;">EMOTION DETECTION USING SPEECH</h2>
  </div>
  """ 
st.markdown(html_temp,unsafe_allow_html=True)
st.subheader('This project can identify the emotion of the person based on the audio sample')
st.subheader('Use this button if you need to record the voice and use it')
audio_bytes = audio_recorder()
if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")
    st.warning('Please download this file and then upload it in the file uploader given below')
st.subheader('Please upload the audio file')
file=st.file_uploader("Choose a file",type=[".wav",".mp3"])
if file is None:
        st.warning('Please upload a valid file!!')
else:
    st.audio(file,format="audio/wav")
    if st.button('Detect emotion'):   
        audio, sample_rate = librosa.load(file) 
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=128)
        mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
        mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
        predicted_label=model.predict(mfccs_scaled_features)
        a=np.argmax(predicted_label,axis=1)
        if a==0:
            out='angry'
        elif a==1:
            out='disgust'
        elif a==2:
            out='fear'
        elif a==3:
            out='happy'
        elif a==4:
            out='neutral'
        elif a==5:
            out='pleasant suprise'
        else:
            out='sad'
        st.success('The detected emotion in the audio input is:- '+str(out))
st.write('DEVELOPED BY V.A.SAIRAM')
st.write('email= sairamadithya2002@gmail.com')
st.write('linkedin= https://www.linkedin.com/in/sairamadithya/')
