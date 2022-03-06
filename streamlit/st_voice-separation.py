# -*- coding: utf-8 -*-
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import nussl
import os
import numpy as np
import tensorflow as tf
from voicesep.core import get_musdb_data
from voicesep.unet import UNetModel
from pydub import AudioSegment
from os import listdir
from os.path import isfile, isdir, join
import shutil


#-------------------------------------
# Définition des fonctions 
#-------------------------------------
@st.cache
# Récupération du listing des musiques 
def get_all_dataviz_csv():
    url = os.path.join('C:\\Users','magla','Documents','GitHub','ProjetDatascientest-VoiceSeparator','data','liste_musdb18_complet.csv')
    url = "https://raw.githubusercontent.com/MickaRiv/ProjetDatascientest-VoiceSeparator/main/data/liste_musdb18_complet.csv"
    return pd.read_csv(url)

def myloss(y_true, y_pred):
  return tf.math.reduce_sum(abs(y_true - y_pred)) + abs(tf.math.reduce_sum(y_true) - tf.math.reduce_sum(y_pred))

#musdb_path = os.path.join('C:\\Users','magla','Documents',"Projet_DataScientest","musdb18")
#unets_path = os.path.join('C:\\Users','magla','Documents',"Projet_DataScientest","UNet")

# Si colab et drive monté
musdb_path = os.path.join("/content","drive","MyDrive","Projet Datascientest","musdb18")
unets_path = os.path.join("/content","drive","MyDrive","Projet Datascientest","UNet")
#-------------------------------------
# Le streamlit 
#-------------------------------------

#from PIL import Image
#image = Image.open(os.path.join("/content","drive","MyDrive","image.jpg"))

#st.image(image, caption='Ceci est une image')

models = [f for f in listdir(unets_path) if isdir(join(unets_path, f))]
model = st.sidebar.radio('Choisissez votre modèle',models)
model_path = os.path.join(unets_path,model)
freq = 8192
window_length = 1023
hop_length = 768
patch_size = 128
nfreq = 512
from tensorflow.keras.models import load_model
unet = load_model(model_path, custom_objects={'myloss': myloss})

stem_ou_mp3 = st.selectbox(
      'Voulez-vous traiter une musique au format stem de la base musdb18 ou une musique de votre choix au format mp3/wav ?',
      ['stem','mp3/wav'])

if stem_ou_mp3 == 'stem':
    train_ou_test = st.selectbox("Voulez-vous une musique de la base train ou test ?",
                                 ['test','train'])
    musdb18path = os.path.join(musdb_path,train_ou_test)
    musdb18songs = [f for f in listdir(musdb18path) if isfile(join(musdb18path, f))]

    titre_musdb = st.selectbox('Choisissez une piste de '+musdb18path,musdb18songs)
    
    musdbsong = os.path.join(musdb18path,titre_musdb)
    musdb18 = 'musdb18_temp'
    if not os.path.exists(musdb18):
        os.mkdir(musdb18)

    shutil.copyfile(musdbsong, os.path.join(musdb18,titre_musdb))
    musdb_demo = get_musdb_data(gather_accompaniment=True,folder=os.getcwd(),subfolder=musdb18)
    
    data = musdb_demo[0]
    os.remove(os.path.join(musdb18,titre_musdb))
    
    "**Le mix**"
    data["mix"].write_audio_to_file('mix.wav')
    st.audio('mix.wav', format='audio/wav')
    

#"**Deuxième option via la commande magique st.write(data['mix'].embed_audio())**"
#st.write(data["mix"].embed_audio())

    "**La vraie voix :**"
    data["sources"]["vocals"].write_audio_to_file('voice.wav')
    st.audio('voice.wav', format='audio/wav')

    signal = data["mix"]

    "**La voix prédite :**"
    unet_separator = UNetModel(signal,unet,freq,window_length,hop_length,patch_size,nfreq)
    audio_pred = unet_separator()[1]
    audio_pred.write_audio_to_file('pred.wav')
    st.audio('pred.wav', format='audio/wav')

    
else:
    uploaded_file = st.file_uploader("Importez une musique",type=['wav','mp3'])

    if uploaded_file is not None:
    #st.write(uploaded_file.name)
        if uploaded_file.name.endswith('wav'):
            audio = AudioSegment.from_wav(uploaded_file)
            file_type = 'wav'
        elif uploaded_file.name.endswith('mp3'):
            audio = AudioSegment.from_mp3(uploaded_file)
            file_type = 'mp3'
            
        "**Le mix :**"
        audio.export(uploaded_file.name, format=file_type)
        st.audio(uploaded_file.name, format=file_type)
        
        signal = nussl.AudioSignal(uploaded_file.name)

        "**La voix prédite :**"
        unet_separator = UNetModel(signal,unet,freq,window_length,hop_length,patch_size,nfreq)
        audio_pred = unet_separator()[1]
        audio_pred.write_audio_to_file('pred.wav')
        st.audio('pred.wav', format='audio/wav')



#-----------------------------------------------------

#st.set_page_config(page_title=None, page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items={"menu":"menu"})
