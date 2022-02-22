import pandas as pd
import streamlit as st

#-------------------------------------
# Pour la table des matières 
#   --> utiliser toc.title("titre")
#-------------------------------------

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
        return f"{self.spaces}- [{self.text}]('#{self.id}')"

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

#-------------------------------------
# Définition des fonctions 
#-------------------------------------
@st.cache
# Récupération du listing des musiques 
def get_dataviz_csv():
    url = "https://raw.githubusercontent.com/MickaRiv/ProjetDatascientest-VoiceSeparator/main/data/liste_musdb18.csv"
    return pd.read_csv(url)

def get_all_scores_csv():
    url = "https://raw.githubusercontent.com/MickaRiv/ProjetDatascientest-VoiceSeparator/main/data/all_scores.csv"
    return pd.read_csv(url)

#-------------------------------------
# Le streamlit 
#-------------------------------------

toc.title("Projet DataScientest - Séparation de voix")

toc.header("Introduction")

st.markdown("**On peut écrire en avec st.markdown**")
st.write('**ou faire un st.write**')
"**Et même aller plus vite avec la commande magique**"
st.text_input('On peut aussi écrire comme ça :', 'Exemple text_input')
st.caption("Ceci est du st.caption")
st.text("Ceci est du st.text")
st.code("Ceci est du st.code")
st.latex("Là c'est du latex : \int a x^2\,df")

from PIL import Image
image = Image.open('G:\Mon Drive\image.jpg')

st.image(image, caption='Ceci est une image')

toc.header("Visualisation des données")

toc.subheader("La base musdb18")

"**Visu du dataframe liste_musdb18 du git :**"
df = get_dataviz_csv()
df


toc.subheader("Spectrogrammes")

toc.subheader("Ecoute des musiques")

toc.header("Construction de notre modèle")

toc.header("Comparaison avec modèles de la littérature")

st.write("**Visu du dataframe all_scores du git :**")
df2 = get_all_scores_csv()
df2

toc.generate()

#-----------------------------------------------------

#st.set_page_config(page_title=None, page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items={"menu":"menu"})
