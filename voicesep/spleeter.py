import subprocess as sp
import nussl
import os

from .core import copy_process_streams

class SpleeterModel:
    
    def __init__(self,mix,in_path,out_path):
        self.mix = mix
        self.in_path = in_path
        self.out_path = out_path
        
    def __call__(self):
        self.mix.write_audio_to_file(os.path.join(self.in_path,"mix.wav"))
        self.separate()
        preds = {}
        for stem in ["accompaniment","vocals"]:
            preds[stem] = nussl.AudioSignal()
            preds[stem].load_audio_from_file(os.path.join(self.out_path,"mix",f"{stem}.wav"))
        return [preds["accompaniment"], preds["vocals"]]
    
    def separate(self):
        cmd = ["spleeter", "separate", "-p", "spleeter:2stems", "-o",
               self.out_path, os.path.join(self.in_path,"mix.wav")]
        print("Launching command: ", " ".join(cmd))
        p = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
        copy_process_streams(p)
        p.wait()
        if p.returncode != 0:
            print("Command failed, something went wrong.")