# MaMPy
Mathematical Morphology Python Library

## Structure du dépôt

* ![MaMPyGUIDemoMaxTree.py](MaMPyGUIDemoMaxTree.py) : Demo Max-Tree avec filtre d'ouverture d'air
* ![maxtree.py](maxtree.py) : Implémentation Max-Tree et ouverture d'air

*

## Utilisation

Afin de tester nos algorithmes, nous avons écris un filtre d'ouverture d'aire basé sur un maxtree.
Pour voir si notre ouverture d'air fonctionne correctement, nous avons fait une image synthétique simple contenant des
carrés de différentes tailles:

![](examples/images/area_test_02_inverted.png)

Voici le résultats que nous obtenons.

Avec une ouverture d'air de 50, rien est filtré puisque aucune composante a une aire < 50.
![](results/area_test_02_inverted_50.png)
Avec une ouverture d'air de 1000.
![](results/area_test_02_inverted_1000.png)
Avec une ouverture d'air de 3000.
![](results/area_test_02_inverted_3000.png)
Avec une ouverture d'air de 6000.
![](results/area_test_02_inverted_6000.png)
