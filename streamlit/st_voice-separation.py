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
from tensorflow.keras.models import load_model
from voicesep.demucs import DemucsModel
from voicesep.spleeter import SpleeterModel
from voicesep.openunmix import OpenUnmixModel

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
#%% dev de la sidebar avec chargement du modèle --------------------------
from PIL import Image
image = Image.open("image.jpg")
st.sidebar.image(image, caption='Ceci est une image')

# pour lister tous les modèles
#models = [f for f in listdir(unets_path) if isdir(join(unets_path, f))]
#model = st.sidebar.radio('Choisissez votre modèle',models)

# on charge 



quel_modele = st.sidebar.selectbox("Modèle :",
                              ["UNet 8 kHz","UNet 4 kHz","Demucs","Repet"])
#Spleeter downgrade tensorflow et OpenUnmix est long

if quel_modele=="UNet 8 kHz" or quel_modele=="UNet 4 kHz":
    if quel_modele=="UNet 8 kHz" :
        model_8kHz = True
        model = st.sidebar.radio("Sélection du modèle :",
                             ["model_20220101_init",
                              "model_20220202_quick",
                              "model_20220127_long_train_sd+ds"])
        freq = 8192
        window_length = 1023
        hop_length = 768
        patch_size = 128
        nfreq = 512

    elif quel_modele=="UNet 4 kHz" :
        model_4kHz = True
        model = st.sidebar.radio("Sélection du modèle :",
                             ["model_20220202_100x2_2.3M_maxepoch",
                              "model_20220206_90x2_2.3M_4k_L2_long",
                              "model_20220206_90x2_2.3M_4k_L2",
                              "model_20220207_aug=8_4k"])
        freq = 4096
        window_length = 511
        hop_length = 384
        patch_size = 64
        nfreq = 256

    model_path = os.path.join(unets_path,model)
    
    if model=="model_20220202_quick" or model=="model_20220127_long_train_sd+ds":
        unet = load_model(model_path, custom_objects={'myloss': myloss})
        
    elif model=="model_20220101_init" or \
        model=="model_20220202_100x2_2.3M_maxepoch" or \
        model=="model_20220206_90x2_2.3M_4k_L2_long" or \
        model=="model_20220206_90x2_2.3M_4k_L2" or \
        model=="model_20220207_aug=8_4k" :
        unet = load_model(model_path)
    
elif quel_modele=="Demucs" or quel_modele=="Spleeter" or quel_modele=="OpenUnmix" or quel_modele=="Repet":
    freq = 16384
    tmp_path = "tmp"
    if not os.path.exists(tmp_path):
        os.mkdir(tmp_path)
    in_path = os.path.join(tmp_path,"input")
    if not os.path.exists(in_path):
        os.mkdir(in_path)
    out_path = os.path.join(tmp_path,"output")
    if not os.path.exists(out_path):
        os.mkdir(out_path)
#%% ------------------------------------------
stem_ou_mp3 = st.selectbox(
      'Format de la musique à traiter : stem de la base musdb18 ou mp3/wav de votre choix :',
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
    
    col1, col2, col3 = st.columns(3)

    with col1:
        "**Le mix**"
        data["mix"].write_audio_to_file('mix.wav')
        st.audio('mix.wav', format='audio/wav')
        fig = plt.figure(figsize=(12, 6))
        nussl.utils.visualize_spectrogram(data["mix"], y_axis='mel')
        plt.ylim(0,freq/2)
        st.pyplot(fig)

    with col2:
        "**La vraie voix :**"
        data["sources"]["vocals"].write_audio_to_file('voice.wav')
        st.audio('voice.wav', format='audio/wav')
        fig = plt.figure(figsize=(12, 6))
        nussl.utils.visualize_spectrogram(data["sources"]["vocals"], y_axis='mel')
        plt.ylim(0,freq/2)
        st.pyplot(fig)

    with col3:
        "**La voix prédite :**"
        signal = data["mix"]
        if quel_modele=="UNet 8 kHz" or quel_modele=="UNet 4 kHz":
            unet_separator = UNetModel(signal,unet,freq,window_length,hop_length,patch_size,nfreq)
        elif quel_modele=="Demucs":
            unet_separator = DemucsModel(signal,"mdx_extra_q",in_path,out_path)
        elif quel_modele=="Spleeter":
            unet_separator = SpleeterModel(signal,in_path,out_path)
        elif quel_modele=="OpenUnmix":
            unet_separator = OpenUnmixModel(signal,"umx")
        elif quel_modele=="Repet":
            unet_separator = nussl.separation.primitive.Repet(signal)

        audio_pred = unet_separator()[1]
        audio_pred.write_audio_to_file('pred.wav')
        st.audio('pred.wav', format='audio/wav')
        fig = plt.figure(figsize=(12, 6))
        nussl.utils.visualize_spectrogram(audio_pred, y_axis='mel')
        st.pyplot(fig)
    

#"**Deuxième option via la commande magique st.write(data['mix'].embed_audio())**"
#st.write(data["mix"].embed_audio())

    
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
            
        col1, col2 = st.columns(2)

        with col1:
            "**Le mix :**"
            audio.export(uploaded_file.name, format=file_type)
            st.audio(uploaded_file.name, format=file_type)
            signal = nussl.AudioSignal(uploaded_file.name)
            fig = plt.figure(figsize=(12, 6))
            nussl.utils.visualize_spectrogram(signal, y_axis='mel')
            st.pyplot(fig)
        
        with col2:
            "**La voix prédite :**"
            signal = nussl.AudioSignal(uploaded_file.name)
            
            if quel_modele=="UNet 8 kHz" or quel_modele=="UNet 4 kHz":
                unet_separator = UNetModel(signal,unet,freq,window_length,hop_length,patch_size,nfreq)
            elif quel_modele=="Demucs":
                unet_separator = DemucsModel(signal,"mdx_extra_q",in_path,out_path)
            elif quel_modele=="Spleeter":
                unet_separator = SpleeterModel(signal,in_path,out_path)
            elif quel_modele=="OpenUnmix":
                unet_separator = OpenUnmixModel(signal,"umx")
            elif quel_modele=="Repet":
                unet_separator = nussl.separation.primitive.Repet(signal)
   
            audio_pred = unet_separator()[1]
            audio_pred.write_audio_to_file('pred.wav')
            st.audio('pred.wav', format='audio/wav')
            fig = plt.figure(figsize=(12, 6))
            nussl.utils.visualize_spectrogram(audio_pred, y_axis='mel')
            st.pyplot(fig)
           


#-----------------------------------------------------

#st.set_page_config(page_title=None, page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items={"menu":"menu"})
