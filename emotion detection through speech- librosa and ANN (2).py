#!/usr/bin/env python
# coding: utf-8

# In[21]:


get_ipython().run_cell_magic('writefile', 'emotion_detection.py', 'import streamlit as st\nimport numpy as np\nimport librosa\nimport tensorflow as tf\nfrom tensorflow import keras\nfrom audio_recorder_streamlit import audio_recorder\nmodel=tf.keras.models.load_model(r"C:\\Users\\sairam\\Downloads\\emotion detection through audio-2.9.h5")\nhtml_temp = """ \n  <div style="background-color:pink ;padding:10px">\n  <h2 style="color:white;text-align:center;">EMOTION DETECTION USING SPEECH</h2>\n  </div>\n  """ \nst.markdown(html_temp,unsafe_allow_html=True)\nst.subheader(\'This project can identify the emotion of the person based on the audio sample\')\nst.subheader(\'Use this button if you need to record the voice and use it\')\naudio_bytes = audio_recorder()\nif audio_bytes:\n    st.audio(audio_bytes, format="audio/wav")\n    st.warning(\'Please download this file and then upload it in the file uploader given below\')\nst.subheader(\'Please upload the audio file\')\nfile=st.file_uploader("Choose a file",type=[".wav",".mp3"])\nif file is None:\n        st.warning(\'Please upload a valid file!!\')\nelse:\n    st.audio(file,format="audio/wav")\n    if st.button(\'Detect emotion\'):   \n        audio, sample_rate = librosa.load(file) \n        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=128)\n        mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)\n        mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)\n        predicted_label=model.predict(mfccs_scaled_features)\n        a=np.argmax(predicted_label,axis=1)\n        st.write(a)\n        if a==0:\n            out=\'angry\'\n        elif a==1:\n            out=\'disgust\'\n        elif a==2:\n            out=\'fear\'\n        elif a==3:\n            out=\'happy\'\n        elif a==4:\n            out=\'neutral\'\n        elif a==5:\n            out=\'pleasant suprise\'\n        else:\n            out=\'sad\'\n        st.success(\'The detected emotion in the audio input is:- \'+str(out))')

