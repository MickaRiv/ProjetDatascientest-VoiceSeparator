from openunmix.predict import separate
from torch import Tensor
import nussl

class OpenUnmixModel:
    
    def __init__(self,mix,model):
        self.mix = mix
        self.model = model
        
    def __call__(self):
        estimates = separate(Tensor(self.mix.audio_data),
                         self.mix.sample_rate,
                         model_str_or_path=self.model)
        preds = {}
        for stem in ["bass","drums","other","vocals"]:
            preds[stem] = nussl.AudioSignal(audio_data_array=estimates[stem][0].numpy(),
                                            sample_rate=self.mix.sample_rate)
        return [preds["bass"]+preds["drums"]+preds["other"], preds["vocals"]]