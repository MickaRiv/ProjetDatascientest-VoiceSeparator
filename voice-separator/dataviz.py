# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 08:27:17 2021

@author: magla
"""
# fonction test
def test():
    print("2 Ceci teste le .py importé")
    print(os.getcwd())

# retourne train ou test ou "not found"
import os
def train_or_test(track_name):
  def file_in(directory):
    return os.path.exists(os.path.join("/content","drive","MyDrive","Projet Datascientest","musdb18",directory,f"{track_name}.stem.mp4"))
  for directory in ["train","test"]:
    if file_in(directory):
      return directory
  else:
    raise IOError(f"Track '{track_name}' not found")
    
# fonction appliquée au df qui retourne la durée en secondes des titres dans une série
import librosa
def track_duration(track_info):
  file = os.path.join("/content","drive","MyDrive","Projet Datascientest","musdb18",track_info["Dataset"],f"{track_info['Track name']}.stem.mp4")
  return librosa.get_duration(filename=file)

# pour les écoutes séparées
# from https://nussl.github.io/docs/tutorials/evaluation.html
import matplotlib.pyplot as plt
from nussl import utils,play_utils
def visualize_and_embed(sources):
    plt.figure(figsize=(10, 7))
    plt.subplot(211)
    utils.visualize_sources_as_masks(
        sources, db_cutoff=-60, y_axis='mel')
    plt.subplot(212)
    utils.visualize_sources_as_waveform(
        sources, show_legend=False)
    plt.tight_layout()
    plt.show()

    play_utils.multitrack(sources, ext='.wav')