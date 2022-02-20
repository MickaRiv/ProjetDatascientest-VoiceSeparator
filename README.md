# Projet Data Scientest - Séparation de voix #

## Contributeurs
* Ephi
* Mickaël
* Stany

## Contexte
Ce projet s'insère dans la formation Data Scientest : l'objectif est d'établir un modèle qui permet d'isoler la voix dans une musique.

## Architecture
Ce dossier contient 3 répertoires :
  1. **data** contient quelques données légères comme le listing des musiques utilisées et les scores finaux des différentes méthodes. Par souci d'espace disque sur git, les sources audio sont à télécharger en ligne.
  2. **notebooks** contient les notebooks python développés dans le cadre du projet. Une rapide description  de chacun d'eux est donnée plus bas.
  3. **voicesep** est notre librairie python, dont nous faisons usage dans les notebooks après installation et import.
  4. **doc** contient les rapports rendus dans le cadre de la formation (présentation du sujet, analyse des données, rapport final).

## Obtention des données audio
La librairie python ```musdb``` permet de télécharger des extraits de 7 secondes de chaque musique, mais pour disposer des musiques entières il est nécessaire de :
* télécharger la base _musdb18.zip_ sur [Zenodo](https://zenodo.org/record/1117372)
* la dézipper :
    * en local si vous utilisez vos ressources
    * sur un drive si vous utilisez les ressources de Google Colab
> **Il faudra adapter les notebooks à votre utilisation, i.e. connecter votre drive si nécessaire et donner le chemin de la base de données musdb18**
 
## Exécution des notebooks
Le bon fonctionnement des notebook requiert de bien suivre la structure définie :
* installation des librairies nécessaires (dont notre librairie voicesep),
* import de tous les paquets python requis,
* préparation de l'environnement de travail (montage du drive, choix des paths, ...),
* lancement des calculs.

## Description des notebooks
* `dataviz.ipynb` : exploration, visualisation et écoute de la base données musdb (7 secondes) et du fichier liste_musdb18.csv.
* `benchmark.ipynb` : tests, visualisations et comparaison rapide des modèles trouvés dans la littérature sur un extrait musical.
* `U_net.ipynb` : construction et entraînement d'un modèle UNet avec générateur, sans augmentation de données. L'accélérateur matériel GPU est activé sur Google Colab.
* `U_net_sans_generateur.ipynb` : construction et entraînement d'un modèle UNet sans générateur, avec augmentation de données. L'accélérateur matériel GPU est activé sur Google Colab.
* `benchmark_complete.ipynb` : calcul des scores et comparaison des modèles de la littérature et des UNet construits sur la base de test complète.
