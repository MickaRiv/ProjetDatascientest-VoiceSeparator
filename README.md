# Projet Data Scientest - Séparation de voix #

## Contributeurs
* Ephi
* Mickaël
* Stany

## Contexte
Ce projet s'insère dans la formation Data Scientest : l'objectif est d'établir un modèle qui permet d'isoler la voix dans une musique.

## Architecture
Ce dossier contient 3 répertoires :
  1. **data** contient les données, en l'occurence le listing des musiques utilisées pour l'apprentissage/validation. Par souci d'espace disque sur git, les sources audio sont à télécharger en ligne
  2. **notebooks** contient les python notebooks développés dans le cadre du projet, leur descrition est donnée plus bas
  3. **voicesep** est la librairie des fonctions python appelées dans les notebooks via ```import voicesep``` au préalable
  4. **doc** contient les 2 rapports rendus dans le cadre de la formation

## Obtention des données audio
La librairie python ```musdb``` permet de télécharger des extraits de 7 secondes de chaque musique, mais pour disposer des musiques entières il est nécessaire de :
* télécharger la base _musdb18.zip_ sur [Zenodo](https://zenodo.org/record/1117372)
* la dézipper :
    * en local si vous utilisez vos ressources
    * sur un drive si vous utilisez les ressources de Google Colab
> **Il faudra adapter les notebooks à votre utilisation, i.e. connecter votre drive si nécessaire et donner le chemin de la base de données musdb18**
 
## Exécution des notebooks sur Google Colab
Les notebooks ayant été développés via Google Colab, vous trouverez au début de chaque notebook l'installation des paquets python nécessaires grâce au fichier setup.py. Voici les paquets nécessaires : *????????????*
* ```pip install nussl```

## Description des notebooks
* dataviz.ipynb : exploration, visualisation et écoute de la base données musdb (7 secondes) et du fichier liste_musdb18.csv
* benchmark.ipynb : tests, visualisations et comparaison rapide des modèles trouvés dans la littérature sur un extrait musical
* models.ipynb : construction de modèles... ??? *--> Classification, à retirer ?*
* U_net.ipynb : 
* U_net_sans_generateur.ipynb :
* metrics.ipynb : comparaison de tous les modèles
* benchmark_complete.ipynb : calcul des scores et comparaison des modèles de la littérature et des UNet construits sur la base de test complète
