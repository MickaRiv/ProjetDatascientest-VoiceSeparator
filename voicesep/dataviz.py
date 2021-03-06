# commentaire
import os
import librosa
import matplotlib.pyplot as plt
from nussl import utils, play_utils

path_to_musdb18 = ""

def test():
    print("2 Ceci teste le .py importé")
    print(os.getcwd())

def file_in_dir(track_name, directory):
    return os.path.exists(os.path.join(path_to_musdb18,
                                     directory,
                                     f"{track_name}.stem.mp4"))

def train_or_test(track_name):
    for directory in ["train","test"]:
        if file_in_dir(track_name, directory):
            return directory
# si on est là c'est qu'on n'a pas fait return
    raise IOError(f"Track '{track_name}' not found")
    
def track_duration(track_info):
    file = os.path.join(path_to_musdb18,track_info["Dataset"],f"{track_info['Track name']}.stem.mp4")
    return librosa.get_duration(filename=file)

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

    play_utils.multitrack(sources)
