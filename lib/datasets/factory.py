from datasets.text import icdar2015ch4

def get_imdb(dataset):
    imdb = icdar2015ch4(dataset)
    return imdb

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
