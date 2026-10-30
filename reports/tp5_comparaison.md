# Modele manuel ou modele AutoML ? (TP05)

| Critere                    | Manuel (rf 500/16)               | AutoML (FLAML 90 s)                          |
|----------------------------|----------------------------------|----------------------------------------------|
| Famille de modele          | foret aleatoire                  | xgboost (XGBClassifier)                      |
| F1 sur le jeu de test      | 0.8669                           | 0.8561                                       |
| Temps de recherche machine | 4.2 s (un seul entrainement)     | 90.3 s (duree_recherche_s)                   |
| Temps humain (estimation)  | ~2 h (reglage manuel au TP2)     | ~5 min (lancer le script)                    |
| Latence mediane (ms)       | 145.6 / 127.0                    | 16.4 / 18.8                                  |
| Latence p95 (ms)           | 388.5 / 286.1                    | 46.8 / 33.6                                  |
| Taille du fichier (Mo)     | 13.37                            | 0.28                                         |
| Explicabilite              | importance des variables, 500 arbres, profondeur 16 | importance des variables, 322 arbres, 4 feuilles max par arbre |

Latences : deux mesures de 200 appels par service (`src/client.py`), en local.

## Decision (5 lignes) : quel modele mettre en production, et pourquoi ?

Nous mettons en production le modele AutoML (XGBoost, `fraude-automl`).
Son F1 (0.8561) n'est inferieur que d'environ 0.01 a celui du modele manuel (0.8669).
En contrepartie, il repond environ 8 fois plus vite (mediane ~17 ms contre ~136 ms, p95 ~40 ms contre ~340 ms).
Il est aussi 48 fois plus leger (0.28 Mo contre 13.4 Mo), ce qui compte pour l'image Docker et le mobile.
Il a ete obtenu en 90 s de calcul, sans reglage humain ; ses 322 petits arbres restent explicables par l'importance des variables.

## Reponses aux questions

### Question 1 : Le choix de production
Nous deployons le modele AutoML. Les criteres retenus en plus du F1 sont :
- la latence : mediane ~17 ms contre ~136 ms, et une queue de distribution (p95) bien plus courte ;
- la taille : 0.28 Mo contre 13.4 Mo, donc un deploiement, un chargement et un passage a l'echelle moins couteux ;
- le cout humain : 90 s de recherche automatique, a comparer aux heures de reglage manuel du TP2 ;
- l'explicabilite, equivalente : dans les deux cas un ensemble d'arbres, avec importance des variables.

L'ecart de F1 (-0.011) est faible au regard de ces gains. Il faut toutefois le surveiller : pour la fraude, chaque faux negatif a un cout.

### Question 2 : Le cout d'un point de terminaison cloud
Un point de terminaison Vertex AI est facture a l'heure de noeud tant qu'un modele y est deploye, meme sans requete.
Pour la plus petite machine proposee (n1-standard-2, region europe-west1), le tarif est de l'ordre de 0.10 $ de l'heure.
Un mois revient donc a environ 0.10 x 24 x 30 ≈ 70 a 80 $, sans aucune requete.
A verifier sur la page de tarification de Vertex AI, date de consultation : ______ (a completer).

### Question 3 : Latence locale et latence cloud
Nos mesures locales (17 a 140 ms en mediane) ne comprennent presque que le temps de calcul du modele et la serialisation JSON, sur la boucle locale 127.0.0.1.
En cloud s'ajoutent :
- l'aller-retour reseau sur Internet (dizaines de millisecondes) ;
- la negociation TLS ;
- l'authentification (jeton OAuth) ;
- le passage par un equilibreur de charge et une passerelle d'API ;
- une machine partagee souvent plus petite que notre poste ;
- un eventuel demarrage a froid.

Le p95 en est le plus affecte. Nos mesures montrent deja que la queue (p95) varie beaucoup d'une execution a l'autre.

### Question 4 : Le moteur change de version
Si le fournisseur change la version de son moteur AutoML, un nouvel entrainement sur les memes donnees peut donner un modele different : autre famille, autres reglages, F1 et latence differents, voire un format d'export ou une API differents.
Le risque est triple :
- perte de reproductibilite : on ne peut plus regenerer le modele en production ;
- regression silencieuse des performances ;
- dependance (verrouillage) vis-a-vis du fournisseur.

Pour s'en premunir, on fige les versions, on conserve les artefacts, et on valide toute nouvelle version sur un jeu de test fixe avant de la promouvoir.

### Question 5 : Reversibilite
- Exporter le modele dans un format ouvert et portable (par exemple un export TensorFlow SavedModel ou un conteneur, cote Vertex), puis le stocker hors de la plateforme.
- L'enregistrer dans notre propre registre (MLflow), avec ses metriques, sa signature et ses dependances, comme dans ce TP.
- Conserver les donnees d'entrainement et le jeu de test versionnes, pour pouvoir re-entrainer ailleurs (FLAML ou scikit-learn) et comparer.
- Servir le modele derriere un alias (`production`) : un retour arriere, ou un changement de fournisseur, se fait en deplacant l'alias, sans modifier le client.

### Question 6 : skops ou cloudpickle
skops n'accepte que des types explicitement juges surs (`skops_trusted_types`). Il convient aux modeles scikit-learn purs comme la foret aleatoire du TP2. Le modele AutoML est un objet XGBoost (ou LightGBM), que skops ne sait pas serialiser de maniere fiable : on utilise donc cloudpickle, qui sait serialiser n'importe quel objet Python.

Le risque de pickle : charger un fichier pickle execute du code arbitraire. Un fichier modele piege pourrait donc compromettre la machine qui le charge.

C'est acceptable ici parce que :
- le modele est produit par nous-memes, sur notre poste ;
- il est stocke dans notre propre registre MLflow local, et n'est charge que par notre service local ;
- aucun fichier ne provient d'une source non fiable.

En production, il faudrait controler l'acces au registre et l'integrite des artefacts (signature, empreinte).
