# MaMPy
Mathematical Morphology Python Library

## Structure du dépôt

* ![MaMPyGUIDemoMaxTree.py](MaMPyGUIDemoMaxTree.py) : Demo Max-Tree avec filtre d'ouverture d'air.
* ![maxtree.py](maxtree.py) : Implémentation Max-Tree et ouverture d'air.
* ![maxtree_profile.py](maxtree_profile.py) : Analyse de performances et de résultats de maxtree.
* ![maxtree_test.py](maxtree_test.py) : Test de maxtree.

* ![doc](doc) : Dossier contenant la documentation et les articles utilisés pour le développement.
* ![results](results) : Dossier contenant les résultats générer avec maxtree_profilel
* ![examples/images](examples/images) : Dossier contenant les images d'entrée utilisé pour les tests et exemples.


## Utilisation

* Exemple d'utilisation de MaMPyGUIDemoMaxTree  
[![MaMPy : max-tree & ouverture d'aire](https://img.youtube.com/vi/4NmaUEBlBkI/0.jpg)](https://youtu.be/4NmaUEBlBkI)

## Maxtree: explications

Le *maxtree* ou l'arbre maximal est un arbre représentant la hiéarchie entre les composantes d'une image. Il peut être
utilisé pour diverses opérations comme une ouverture d'aire.

Avant d'éxpliquer comment un arbre maximal est créé, il faut comprendre comment on le représente car la structure 
proposée dans l'article aide beaucoup à la création de celui-ci.

Le maxtree est représenté à l'aide d'un tableau (nommé *parent* dans l'article; de même dimension que l'image) qui 
donne pour chaque élement un lien vers son parent.
De plus, on a un autre tableau de dimension 1 (nommé *S*) qui contient les noeuds triés selon leur hauteur dans l'arbre. 
Celui-ci permet de facilement parcourir l'arbre.
Avec ces deux informations, on a une réprésentation complète de l'arbre maximal et on peut écrire des filtres.

Il existe plusieurs techniques pour créer le *maxtree*, nous avons implémenté des algorithmes dit *Immersion algorithms* 
et un algorithme dit *Flooding Algorithm* qui est une extension des algorithmes par immersion. 
Le dernier est celui que nous utilisons car c'est le plus performant.

Les algorithmes par immersion sont simples, ils sont constitués de 2 étapes:
1. Triage des élements de l'image en entrée selon leurs valeurs
2. Fusion des élements selon leurs valeurs pour former l'arbre

Le tri des élements est trivial et nous permet d'obtenir directement *S*. On parcoure ensuite chaque élement de *S* en 
partant des feuilles et on définit l'élement courant comme le parent de ses voisins si ceux-ci sont des élements ayant déjà étés traités.

Nous n'avons pas besoin de vérifier les valeurs des élements car on utilise *S* pour les parcourir et ils sont triés.
Contre-intuitivement, des élements étant dans la même composante (même valeure et même zone) ne sont pas dans le même noeud.
En effet, chaque élement est son propre noeud (ce sont des singletons). On utilise la notion de canonique pour savoir si un noeud donné fait parti 
d'une composante ou non. Un noeud est dit canonique si la valeure de son parent est différente sa propre valeur.

Exemple avec une image tirée de l'article:
![](doc/maxtree_representation.png)

Ici, les noeuds **E, H et D** font parti de la même composante. C'est le même noeud mais la représentation choisie ne 
permet pas de combiner les noeuds directement. **E** est canonique alors que **H et D** ne le sont pas.

L'algorithme par **Flooding** suit la même idée sauf qu'il fonctionne par propagation pour économiser des instructions. 

