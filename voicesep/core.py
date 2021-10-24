from nussl import datasets

def get_musdb_data(gather_accompaniment=False):
    if gather_accompaniment:
        tfm = datasets.transforms.SumSources([['drums', 'bass', 'other']])
    else:
        tfm = None
    return datasets.MUSDB18(download=True, transform=tfm)