# Max Tree

Afin de tester nos algorithmes, nous avons écris un filtre d'ouverture d'aire basé sur un maxtree.
Pour voir si notre ouverture d'air fonctionne correctement, nous avons fait une image synthétique simple contenant des
carrés de différentes tailles :
![](../examples/images/area_test_02_inverted.png)  
Afin de tester les performances de nos algorithmes, nous avons essayé le filtre d'ouverture d'aire sur une image plus
grande : Noyau_Slice68 de taille 1536 x 2048.
![](../examples/images/Noyau_Slice68.png)


Résultats :

Avec une ouverture d'air de 50, rien n'est filtré puisque aucune composante a une aire < 50.
![](area_test_02_inverted_50.png)
Avec une ouverture d'air de 1000.
![](area_test_02_inverted_1000.png)
Avec une ouverture d'air de 3000.
![](area_test_02_inverted_3000.png)
Avec une ouverture d'air de 6000.
![](area_test_02_inverted_6000.png)


Noyau_Slice68 avec une ouverture d'air de 20000
![](Noyau_Slice68_20000.png)
