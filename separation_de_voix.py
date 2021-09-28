# à ne faire qu'une fois pour installer le paquet nussl
#!pip install nussl

# Chargement des paquets
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import os

#------------------------------------
# Préparation des données
#------------------------------------

# pour fonctionner dans google collab
from google.colab import drive
# montage du compte drive
drive.mount('/content/drive')

# Lecture du csv contenant le listing des musiques
url='https://docs.google.com/spreadsheets/d/e/2PACX-1vSYacI0OMN9buRCXZRBrIKgGg6NbcP1HXMDbNElVidLn72JJWs-0sSImGjUedBWKURR5bs2_F_mjWi0/pub?gid=0&single=true&output=csv'
df=pd.read_csv(url)
df.head()

# Correction de quelques coquilles de frappe dans le csv
df = df.replace('Jokers Jacks & Kings - Sea Of Leaves','Jokers, Jacks & Kings - Sea Of Leaves')
df = df.replace('Patrick Talbot - Set Free Me','Patrick Talbot - Set Me Free')

# fonction pour localiser un titre de musdb18
# retourne train ou test ou "not found"
def train_or_test(track_name):
  def file_in(directory):
    return os.path.exists(os.path.join("drive","MyDrive","Projet Datascientest","musdb18",directory,f"{track_name}.stem.mp4"))
  for directory in ["train","test"]:
    if file_in(directory):
      return directory
  else:
    raise IOError(f"Track '{track_name}' not found")

# fonction appliquée au df qui retourne la durée en secondes des titres dans une série
def track_duration(track_info):
  file = os.path.join("drive","MyDrive","Projet Datascientest","musdb18",track_info["Dataset"],f"{track_info['Track Name']}.stem.mp4")
  return librosa.get_duration(filename=file)

# création de 2 colonnes grâce aux 2 fonctions précédentes
# Dataset : train/test
# Duration : durée de la musique
df["Dataset"] = df["Track Name"].apply(train_or_test)
df["Duration"] = df.apply(track_duration,axis=1)
df

#------------------------------------
# premières visualisations des données
#------------------------------------
# nombre de musiques en fonction de leur genre
plt.figure(figsize=(12,6))
sns.countplot(df['Genre'],order=df['Genre'].value_counts().index)
plt.show()

# nombre de musiques en fonction de leur genre et de leur set (train ou test)
plt.figure(figsize=(12,6))
sns.countplot(df['Genre'],order=df['Genre'].value_counts().index,hue=df["Dataset"])
plt.show()

# nombre de musiques en fonction de leur durée avec distribution
sns.histplot(df['Duration'],kde=True)
plt.show()

# nombre de musiques en fonction de leur durée et de leur set (train ou test)
sns.histplot(df['Duration'][df["Dataset"]=="train"],label="train")
sns.histplot(df['Duration'][df["Dataset"]=="test"],label="test",color="orange")
plt.legend()
plt.show()

# Affichage des 10 musique les plus courtes
df.sort_values(by="Duration").head(10)

# quelques stats sur l'échantillon de test
# seule la durée est numérique donc on ne voit qu'elle
df[df["Dataset"]=="test"].describe()

# quelques stats sur l'échantillon d'entrainement train
# seule la durée est numérique donc on ne voit qu'elle
df[df["Dataset"]=="train"].describe()

#------------------------------------
# exploration/analyse des données
#------------------------------------
import nussl
from IPython.display import Audio
from IPython.display import display as AudioDisplay

# MUSDB18 est déjà dans le paquet nussl, 
# on s'appuie donc sur les fonctions de ce paquet

# création d'un paquet musdb avec 7s secondes de chaque musique de la base musdb18
# sera dans /root/.nussl/musdb18
musdb = nussl.datasets.MUSDB18(download=True)

# on importe ce paquet, on stocke (télécharge) dans mus on regarde le nombre de musiques
# root/MUSDB18/MUSDB18-7
# 144 dans nussl au lieu de 150 dans ce que DataScientest
# les 6 manquantes sont celles qui durent moins de 30 secondes et
# qui s'appellent ...Delta...
import musdb as musdb_package
mus = musdb_package.DB(download=True)
len(mus)

# on imprime les titres et numéros de musique dont le titre contient "elta"
print([(i,mus[i].name,"\n") for i in range(len(mus)) if ("elta" in mus[i].name)])

# écoute des titres pour en sélectionner quelques uns
display_audio_mix = True
display_audio_sources = True
i = 54#54  #  stany : j'en choisis un avec une voix assez présente avec pas mal d'harmoniques

# la musique mélangée (mix)
if display_audio_mix:
  AudioDisplay(Audio(data=musdb[i]['mix'].audio_data, rate=musdb[i]['mix'].sample_rate))
  
# la musique décomposée par source (drums, bass, others, vocals)
if display_audio_sources:
  for source,source_data in musdb[i]['sources'].items():
    print(source)
    AudioDisplay(Audio(data=source_data.audio_data, rate=source_data.sample_rate))
    
# Visualisation des données audio de i
#--------------------------------------

# Signal temporel du mix
print("mix")
plt.figure(figsize=(10,8))
plt.subplot(311)
nussl.core.utils.visualize_sources_as_waveform({"Mix": musdb[i]['mix']})

# Signaux temporels des 4 sources superposées
plt.subplot(312)
nussl.core.utils.visualize_sources_as_waveform(musdb[i]['sources'])

# Signaux temporels de la voix et de l'accompagnement superposés
plt.subplot(313)
nussl.core.utils.visualize_sources_as_waveform({"Vocal":musdb[i]['sources']["vocals"],"Accompagnement":musdb[i]['sources']["drums"]+musdb[i]['sources']["bass"]+musdb[i]['sources']["other"]})
plt.tight_layout()
plt.show()

# spectogrammes
plt.figure(figsize=(12,4))

# spectogramme du mix en fréquence
# stany : je mets en log pour être comparatif freq/mel et une légende en dB
plt.subplot(121)
nussl.core.utils.visualize_spectrogram(musdb[i]['mix'],y_axis='log')
plt.colorbar(format='%+2.0f dB')

# spectogramme du mix en mel
plt.subplot(122)
nussl.core.utils.visualize_spectrogram(musdb[i]['mix'],y_axis="mel")
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()

# spectogramme du mix en fréquence, échelle linéaire
# stany : spectr total/voix/acc - zoom fréquence pour bien voir la voix
plt.figure(figsize=(12,6))
plt.subplot(311)
nussl.core.utils.visualize_spectrogram(musdb[i]['mix'],y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.ylim(0,2000)
plt.title('Mélange')

# spectogramme de la voix en fréquence, échelle linéaire
plt.subplot(312)
nussl.core.utils.visualize_spectrogram(musdb[i]['sources']["vocals"],y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.ylim(0,2000)
plt.title('Voix')

# spectogramme de l'accompagnement en fréquence, échelle linéaire
plt.subplot(313)
nussl.core.utils.visualize_spectrogram(musdb[i]['sources']["drums"]+musdb[i]['sources']["bass"]+musdb[i]['sources']["other"],y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.ylim(0,2000)
plt.title('Accompagnement')
plt.tight_layout()
plt.show()

# visualisation de la transformée de Fourrier du signal temporel du mix
from scipy import fft
print(musdb[i]["mix"].__dict__)
nussl.core.utils.visualize_sources_as_waveform({"Somme":musdb[i]["mix"]})
plt.show()
fft(musdb[i]["mix"]._audio_data[0][::10])

# alors là je suis perdue...
from scipy.fft import fft, fftfreq
import numpy as np
# Number of sample points
N = 600
# sample spacing
T = 1.0 / 800.0
x = np.linspace(0.0, N*T, N, endpoint=False)
y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
yf = fft(y)
xf = fftfreq(N, T)[:N//2]
import matplotlib.pyplot as plt
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.grid()
plt.show()

# identification de la voix qui chante ou qui s'arrête
# Stany  : Masque voix
# On commence par trouver un morceau avec des interruptions voix (e.g. 59, 70)

i = 59 

#écoute...
AudioDisplay(Audio(data=musdb[i]['mix'].audio_data, rate=musdb[i]['mix'].sample_rate))
for source,source_data in musdb[i]['sources'].items():
    print(source)
    AudioDisplay(Audio(data=source_data.audio_data, rate=source_data.sample_rate))

# spectro mix/voix
plt.figure(figsize=(12,6))
plt.subplot(211)
nussl.core.utils.visualize_spectrogram(musdb[i]['mix'],y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.ylim(0,8000)
plt.title('Mélange')
plt.subplot(212)
nussl.core.utils.visualize_spectrogram(musdb[i]['sources']["vocals"],y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.ylim(0,8000)
plt.title('Voix')
plt.tight_layout()
plt.show()

# Masque voix. On fait bourrin : seuil sur niveau  de psd du spectrogramme...

import numpy as np
i=59
voice=musdb[i]['sources']["vocals"]
stft = voice.stft()

# coupure en db (par rapport au max)
db_cutoff = -20. 

# niveau puissance en db (en relatif par rapport max.)
psd = 10*np.log10(voice.power_spectrogram_data/voice.power_spectrogram_data.max())

# je moyenne sur canaux et on chope le max dans toute la gamme fréquentielle
psdmax=np.max( (psd[:,:,0]+psd[:,:,1])/2., axis=0)

# puis masque basé sur cette valeur
tmask=np.where(psdmax  < db_cutoff, 0., 1.)

# temps morceau et taille dt des chunk (nhop) de la stft
t=voice.audio_data.shape[1]/voice.sample_rate 
nhop=mask.shape[1]
dt=t/nhop
t= dt*np.arange(nhop) 

# on trace... 0= pas de voix
plt.figure(figsize=(12,6))
plt.subplot(211)
nussl.core.utils.visualize_spectrogram(voice,y_axis='linear')
plt.title('Spectro voix')
plt.ylim(0,6000)
plt.xlim(0,7)
plt.subplot(212)
plt.title('Masque voix')
plt.scatter(t,y=tmask)
plt.xlim(0,7)
plt.yticks([0,1],['Pas de voix', 'Voix'])
plt.tight_layout()
plt.show()