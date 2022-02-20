import nussl
import librosa
import numpy as np
import os
import concurrent.futures

def preprocess(*music_data,
               freq=8192,
               window_length=1023,
               hop_length=768,
               normalize=True,
               predvoix=False):
  
    mags = []

    for i,data in enumerate(music_data):

        # Resampling (pour data plus light)
        if freq is not None:
            data.resample(freq,res_type='kaiser_fast')

        # Passage en mono
        data.to_mono()

        # fft mix et voix (magnitude normalisée et phase)
        stftdata = data.stft(window_length=window_length,hop_length=hop_length)
        magdata, phasedata = librosa.magphase(stftdata)

        if i==0:
            magfirst = magdata
            phasefirst = phasedata

        mags.append(magdata)

    if normalize:
        norm = magfirst.max()
        for i,mag in enumerate(mags):
            mags[i] /= norm
            
        if predvoix:
            maxtvoice=np.max(mags[1], axis=0)
            maxtvoice=10*np.log(maxtvoice+1e-10)
            db_cutoff=-20
            tmask=np.where(maxtvoice  < db_cutoff, 0, 1)
            return (*mags, phasefirst, norm, tmask)
        else:
            return (*mags, phasefirst, norm)

    return (*mags, phasefirst)

def reshape(*X, patch_size=128, extend=False):

    X_reshaped = []

    if extend:
        nimages = int(np.ceil(X[0].shape[1]/patch_size))
    else:
        nimages = X[0].shape[1]//patch_size
    newsize = nimages*patch_size

    for Xi in X:
        # transposition pour mettre le temps en première composante  
        Xi = Xi.T

        # split en "images" temporelles de taille patch_size (128 dans le papier d'origine)
        if extend:
          Xi_resized = np.concatenate((Xi,np.zeros((newsize-Xi.shape[0],Xi.shape[1]))))
        else:
          Xi_resized = Xi[:newsize]

        Xi = np.array(np.split(Xi_resized, nimages, axis=0))    # découpage 

        Xi = np.expand_dims(Xi,axis=3) # rajout d'une dimension (canal) 

        X_reshaped.append(Xi)

    return tuple(X_reshaped)
  
def get_data(all_data, num, freq, window_length, hop_length, cache_on_disk, cache_path):

    magmix, magvoice, *_ = preprocess(all_data[num]["sources"]["vocals"], all_data[num]["mix"],
                                      freq=freq, window_length=window_length, hop_length=hop_length)
    magmix = magmix[:,:,0]
    magvoice = magvoice[:,:,0]
    if cache_on_disk:
        os.mkdir(cache_path)
        np.save(os.path.join(cache_path,"magmix.npy"), magmix)
        np.save(os.path.join(cache_path,"magvoice.npy"), magvoice)
    return magmix, magvoice
  
def data_generator(all_data,
                   batch_size = 10,
                   nfreq = 512,
                   freq = 8192,
                   window_length = 1023,
                   hop_length = 768,
                   patch_size = 128,
                   cache_on_disk = False,
                   cache_dir = "cache",
                   randomize_batches = True,
                   min_max = None,
                   parallel = True,
                   max_workers = None):
    # Mix as input, voice as output
    if cache_on_disk and not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    magmixes, magvoices = {},{}
    if min_max is None:
        min = 0
        max = len(all_data)
    else:
        min, max = min_max
    my_range = np.arange(min,max)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        workers = {}
        counts = []
        while True:

            if randomize_batches:
                np.random.shuffle(my_range)

            for count,num in enumerate(my_range):

                counts.append(count)
                cache_path = os.path.join(cache_dir,f"{num}_{freq}_{window_length}_{hop_length}")
                if cache_on_disk and os.path.exists(cache_path):
                    magmix = np.load(os.path.join(cache_path,"magmix.npy"))
                    magvoice = np.load(os.path.join(cache_path,"magvoice.npy"))
                    magmixes[count] = magmix
                    magvoices[count] = magvoice
                else:
                    func = get_data
                    args = (all_data, num, freq, window_length, hop_length, cache_on_disk, cache_path)
                    if parallel:
                        workers[count] = executor.submit(func, *args)
                    else:
                        magmix, magvoice = func(*args)
                        magmixes[count] = magmix
                        magvoices[count] = magvoice

                if((count+1)%batch_size == 0 or count+1 == len(my_range)):

                    if parallel:
                        results = {i:w.result() for i,w in workers.items()}
                        for i in results.keys():
                            magmixes[i] = results[i][0]
                            magvoices[i] = results[i][1]
                    X_mix = np.concatenate([magmixes[i] for i in counts], axis=1)
                    X_voice = np.concatenate([magvoices[i] for i in counts], axis=1)
                    X_mix, X_voice = reshape(X_mix, X_voice, patch_size=patch_size)

                    yield X_mix, X_voice

                    X_mix, X_voice = None, None
                    magmixes, magvoices = {}, {}
                    workers = {}
                    counts = []

class UNetModel:
    def __init__(self,mix,model,freq,window_length,hop_length,patch_size,nfreq):
        self.mix = mix
        self.model = model
        self.freq = freq
        self.window_length = window_length
        self.hop_length = hop_length
        self.patch_size = patch_size
        self.nfreq = nfreq
    def __call__(self):
        magmix, phase, norm = preprocess(self.mix,freq=self.freq,window_length=self.window_length,hop_length=self.hop_length)
        X_mix = reshape(magmix[:,:,0], patch_size=self.patch_size, extend=True)
        X_voice_pred = self.model.predict(X_mix)
        magmix_pred = X_voice_pred.reshape(-1,self.nfreq,1).transpose(1,0,2)
        end = magmix.shape[1]
        audio_pred = nussl.AudioSignal(stft=magmix_pred[:,:end]*norm*phase,sample_rate=self.freq)
        audio_pred.istft(window_length=self.window_length,hop_length=self.hop_length)
        audio_pred.truncate_seconds(self.mix.signal_duration)
        return [audio_pred*2,audio_pred]
