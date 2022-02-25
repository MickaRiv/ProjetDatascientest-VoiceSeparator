import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import nussl
import os,sys

#-------------------------------------
# Pour la table des matières 
#   --> utiliser toc.title("titre")
#-------------------------------------
#%% table des matières
class Header:
    tag: str = ""

    def __init__(self, text: str):
        self.text = text

    @property
    def id(self):
        """Create an identifcator from text."""
        return "".join(filter(str.isalnum, self.text)).lower()

    @property
    def anchor(self):
        """Provide html text for anchored header. Example: 
           <h1 id="abcdef">Abc Def</h1>
        """
        return f"<{self.tag} id='{self.id}'>{self.text}</{self.tag}>"

    def toc_item(self) -> str:
        """Make markdown item for TOC listing. Example:
           '  - <a href='#abc'>Abc</a>'
        """
        return f"{self.spaces}- [{self.text}](#{self.id})"

    @property
    def spaces(self):
        return dict(h1="", h2=" " * 2, h3=" " * 4).get(self.tag)

assert Header("abc").spaces is None

class H1(Header):
    tag = "h1"

class H2(Header):
    tag = "h2"

assert H2("Abc").toc_item() == "  - [Abc]('#abc')"

class H3(Header):
    tag = "h3"

class TOC:
    """
    Original code, used with modifications:
    https://discuss.streamlit.io/t/table-of-contents-widget/3470/8?u=epogrebnyak
    """

    def __init__(self):
        self._headers = []
        self._placeholder = st.empty()

    def title(self, text):
        self._add(H1(text))

    def header(self, text):
        self._add(H2(text))

    def subheader(self, text):
        self._add(H3(text))

    def generate(self):
        text = "\n".join([h.toc_item() for h in self._headers])
        self._placeholder.markdown(text, unsafe_allow_html=True)

    def _add(self, header):
        st.markdown(header.anchor, unsafe_allow_html=True)
        self._headers.append(header)


class TOC_Sidebar(TOC):
    def __init__(self):
        self._headers = []
        self._placeholder = st.sidebar.empty()
        
toc = TOC_Sidebar()
#%%
#-------------------------------------
# Définition des fonctions 
#-------------------------------------
@st.cache
# Récupération du listing des musiques 
def get_dataviz_csv():
    url = "https://raw.githubusercontent.com/MickaRiv/ProjetDatascientest-VoiceSeparator/main/data/liste_musdb18.csv"
    return pd.read_csv(url)

def get_all_dataviz_csv():
    url = os.path.join('C:\\Users','magla','Documents','GitHub','ProjetDatascientest-VoiceSeparator','data','liste_musdb18_complet.csv')
    url = "https://raw.githubusercontent.com/MickaRiv/ProjetDatascientest-VoiceSeparator/main/data/liste_musdb18_complet.csv"
    return pd.read_csv(url)


def get_all_scores_csv():
    url = "https://raw.githubusercontent.com/MickaRiv/ProjetDatascientest-VoiceSeparator/main/data/all_scores.csv"
    return pd.read_csv(url)


musdb_path = os.path.join('C:\\Users','magla','Documents',"Projet_DataScientest","musdb18")
unets_path = os.path.join('C:\\Users','magla','Documents',"Projet_DataScientest","UNet")

# Si colab et drive monté
musdb_path = os.path.join("/content","drive","MyDrive","Projet Datascientest","musdb18")
unets_path = os.path.join("/content","drive","MyDrive","Projet Datascientest","UNet")
#-------------------------------------
# Le streamlit 
#-------------------------------------

toc.title("Projet DataScientest - Séparation de voix")

toc.header("Introduction")

st.markdown("**On peut écrire avec st.markdown**")
st.write('**ou faire un st.write**')
"**Et même aller** ```plus vite``` **avec la commande magique**"
text = 'On utilise le markdown pour écrire <span style="color:Green">en couleur</span>'
st.markdown(text,unsafe_allow_html=True)
text = 'Autre <span style="background-color:Blue;color:White">Exemple</span>.'
st.markdown(text,unsafe_allow_html=True)
st.text_input('On peut aussi écrire comme ça :', 'Exemple text_input')
st.caption("Ceci est du st.caption")
st.text("Ceci est du st.text")
st.code("Ceci est du st.code")
st.latex("Là \ c'est \ du\ latex\ :\ \int a x^2\,df")

from PIL import Image
image = Image.open(os.path.join("/content","drive","MyDrive","image.jpg"))

st.image(image, caption='Ceci est une image')

toc.header("Visualisation des données")

toc.subheader("La base musdb18")

"**Visu du dataframe liste_musdb18 du git :**"
df = get_all_dataviz_csv()
df.columns
cols = ['Track name','Genre']
st_ms = st.multiselect('Colonnes',df.columns.to_list(),default=cols)
st_ms

sd = st.selectbox(
    "Sélectionne un Plot",
    [
     "Genre",
     "Durée"
     ])

fig = plt.figure(figsize=(12, 6))

if sd == "Genre":
    sns.countplot(df['Genre'],order=df['Genre'].value_counts().index)
elif sd == "Durée":
    sns.histplot(df['Duration'],kde=True)
st.pyplot(fig)

toc.subheader("Spectrogrammes")

toc.subheader("Ecoute des musiques")

toc.header("Construction de notre modèle")
from voicesep.core import get_musdb_data

#musdb_train = get_musdb_data(gather_accompaniment=True,folder=musdb_path,subfolder="train")
musdb_test = get_musdb_data(gather_accompaniment=True,folder=musdb_path,subfolder="test")

import tensorflow as tf

def myloss(y_true, y_pred):
  return tf.math.reduce_sum(abs(y_true - y_pred)) + abs(tf.math.reduce_sum(y_true) - tf.math.reduce_sum(y_pred))

from tensorflow.keras.models import load_model
model_path = os.path.join(unets_path,"model_20220202_quick")
unet = load_model(model_path, custom_objects={'myloss': myloss})

data = musdb_test[0]

print("Mix:")
plt.close("all")
fig = plt.figure(figsize=(12, 6))
data["mix"].embed_audio()
####### PROBLEME ##################
# st.audio(data["mix"].embed_audio())
####### PROBLEME ##################
nussl.utils.visualize_spectrogram(data["mix"], y_axis='mel')
st.pyplot(fig)


toc.header("Comparaison avec modèles de la littérature")

st.write("**Visu du dataframe all_scores du git :**")
df2 = get_all_scores_csv()
df2

toc.generate()

#-----------------------------------------------------

#st.set_page_config(page_title=None, page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items={"menu":"menu"})
