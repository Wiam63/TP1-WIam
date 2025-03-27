##exercice1##
import pandas as pd

# Création de la Series à partir du  dictionnaire deja donne
serie_exo1 = pd.Series({'a': 100, 'b': 200, 'c': 300})
print('la serie initial  modifications')
print(serie_exo1)

# Modifier la serie
# Ajouter un nouvel élément
serie_exo1['d'] = 400

# Modification de la valeur associée à 'b' (de 200 à 250)
serie_exo1['b'] = 250

print('la serie apres modifications')
print(serie_exo1)

##exercice2##
import pandas as pd

# Création du DataFrame initial avant modification
df_initial = pd.DataFrame({
    'A': [1, 4, 7],
    'B': [2, 5, 8],
    'C': [3, 6, 9]
})

# Affichage du DataFrame initial
print("DataFrame Initial :")
print(df_initial)

# Copie du DataFrame pour modification
df_modifie = df_initial.copy()

# Ajout de la colonne 'D' avec les valeurs [10, 11, 12]
df_modifie['D'] = [10, 11, 12]

# Suppression de la colonne 'B'
df_modifie.drop(columns='B', inplace=True)

# Affichage du DataFrame modifié
print("\nDataFrame Modifié :")
print(df_modifie)

##exercice3##

import pandas as pd

# Création du DataFrame à partir d'un dictionnaire
df_exo3 = pd.DataFrame({
    'A': [1, 4, 7],
    'B': [2, 5, 8],
    'C': [3, 6, 9]
})

# Sélection de la colonne 'B'
col_B = df_exo3['B']

# Sélection des colonnes 'A' et 'C'
cols_A_C = df_exo3[['A', 'C']]

# Sélection de la ligne avec index 1
row_index_1 = df_exo3.loc[1]

print("Sélection de la colonne B:")
print(col_B)

print("\nSélection des colonnes A et C:")
print(cols_A_C)

print("\nSélection de la ligne avec index 1:")
print(row_index_1)

##exercice4##

import pandas as pd
from numpy.random import randn

# Création du DataFrame
df_exo4 = pd.DataFrame({
    'A': [1, 4, 7],
    'B': [2, 5, 8],
    'C': [3, 6, 9]
})

# Ajout de la colonne 'Sum' (somme des colonnes A, B et C)
df_exo4['Sum'] = df_exo4['A'] + df_exo4['B'] + df_exo4['C']
print("\nDataFrame avec la colonne 'Sum'")
print(df_exo4)

# Suppression de la colonne sum
df_exo4.drop(columns='Sum', inplace=True)
print("\nDataFrame apres la suppression de la colonne 'Sum'")
print(df_exo4)

# Ajout d'une colonne 'Random' avec des nombres aléatoires
df_exo4['Random'] = randn(3)

print("\nDataFrame avec la colonne 'Random'")
print(df_exo4)


##exercice5##

import pandas as pd

# Création du DataFrame 'left'
left = pd.DataFrame({
    'key': [1, 2, 3],
    'A': ['A1', 'A2', 'A3'],
    'B': ['B1', 'B2', 'B3']
})

# Création du DataFrame 'right'
right = pd.DataFrame({
    'key': [1, 2, 3],
    'C': ['C1', 'C2', 'C3'],
    'D': ['D1', 'D2', 'D3']
})

# Fusion avec un inner join
merged_inner = pd.merge(left, right, how='inner', on='key')
print("\nFusion avec Inner Join:")
print(merged_inner)

# Modification pour un outer join
merged_outer = pd.merge(left, right, how='outer', on='key')
print("\nFusion avec Outer Join:")
print(merged_outer)

# Ajout d'une nouvelle colonne E au DataFrame right
right['E'] = ['E1', 'E2', 'E3']

# Mise à jour la fusion pour inclure la nouvelle colonne E
merged_with_E = pd.merge(left, right, how='outer', on='key')
print("\nFusion avec Outer Join + Nouvelle Colonne E:")
print(merged_with_E)

##exercice6##

import pandas as pd
import numpy as np

# Création du DataFrame avec des valeurs nan
df_exo6 = pd.DataFrame({
    'A': [1.0, np.nan, 7.0],
    'B': [np.nan, 5.0, 8.0],
    'C': [3.0, 6.0, np.nan]
})

#Remplacement des nan valeurs  par 0
df_replace_0 = df_exo6.fillna(0)
print("\nDataFrame avec nan remplacés par 0:")
print(df_replace_0)

#Remplacement des valeurs NaN par la moyenne de la colonne
df_replace_mean = df_exo6.fillna(df_exo6.mean())
print("\nDataFrame avec NaN remplacés par la moyenne de chaque colonne:")
print(df_replace_mean)

#Suppression des lignes qui contien au moins une valeur nan
df_drop_na = df_exo6.dropna()
print("\nDataFrame après suppression des lignes contenant des NaN:")
print(df_drop_na)


##exercice7##

import pandas as pd

# Création du DataFrame
df_exo7 = pd.DataFrame({
    'Category': ['A', 'B', 'A', 'B', 'A', 'B'],
    'Value': [1, 2, 3, 4, 5, 6]
})
print("DataFrame initial:")
print(df_exo7)
#Groupement par 'Category' et calcul de la moyenne
df_mean = df_exo7.groupby('Category').mean()
print("\nMoyenne par catégorie:")
print(df_mean)

# Modification pour calculer la somme au lieu de la moyenne
df_sum = df_exo7.groupby('Category').sum()
print("\nSomme par catégorie:")
print(df_sum)

#Groupement par 'Category' et comptage du nombre d'entrées dans chaque groupe
df_count = df_exo7.groupby('Category').count()
print("\nNombre d'entrées par catégorie:")
print(df_count)

##exercice8##

import pandas as pd

# Création du DataFrame initial
df_exo8 = pd.DataFrame({
    'Category': ['A', 'A', 'A', 'B', 'B', 'B'],
    'Type': ['X', 'Y', 'X', 'Y', 'X', 'Y'],
    'Value': [1, 2, 3, 4, 5, 6]
})

print("\nDataFrame Initial :")
print(df_exo8)

# Création d'un pivot table affichant la moyenne de 'Value' pour chaque 'Category' et 'Type'
pivot_mean = pd.pivot_table(df_exo8, values='Value', index='Category', columns='Type', aggfunc='mean')
print("\nPivot Table - Moyenne :")
print(pivot_mean)

# Modification du pivot table pour afficher la somme au lieu de la moyenne
pivot_sum = pd.pivot_table(df_exo8, values='Value', index='Category', columns='Type', aggfunc='sum')
print("\nPivot Table - Somme :")
print(pivot_sum)

# Ajout de marges pour afficher les totaux avec la somme
pivot_with_margins_sum = pd.pivot_table(df_exo8, values='Value', index='Category', columns='Type', aggfunc='sum', margins=True)
print("\nPivot Table - Somme avec Marges :")
print(pivot_with_margins_sum)

##exercice9##

import pandas as pd
import numpy as np

#Création d'un DataFrame de série temporelle avec une plage de dates
date_range = pd.date_range(start='2023-01-01', periods=6, freq='D')

# Génération de valeurs aléatoires pour la série temporelle
values = np.random.randn(6)

df_exo9 = pd.DataFrame({'Date': date_range, 'Value': values})

print("\nDataFrame Série Temporelle :")
print(df_exo9)

#Définition de la colonne 'Date' comme index
df_exo9.set_index('Date', inplace=True)

#Resampling des données pour calculer la somme sur des périodes de 2 jours
df_resampled = df_exo9.resample('2D').sum()
print("\nDataFrame Resamplé (Somme sur 2 jours) :")
print(df_resampled)


##exercice10##

import pandas as pd
import numpy as np

# Création du DataFrame avec des valeurs NaN
df_exo10 = pd.DataFrame({
    'A': [1.0, 2.0, np.nan],
    'B': [np.nan, 5.0, 8.0],
    'C': [3.0, np.nan, 9.0]
})

# Suppression des lignes contenant au moins une valeur NaN sur le DataFrame initial
df_drop_na_initial = df_exo10.dropna()

# Interpolation complète dans les deux directions
df_interpolated = df_exo10.interpolate(limit_direction='both')

# Suppression des lignes contenant au moins une valeur NaN sur le DataFrame interpolé
df_drop_na_interpolated = df_interpolated.dropna()

# Affichage des DataFrames
print("\nDataFrame Initial:")
print(df_exo10)

print("\nDataFrame après suppression des NaN (Initial):")
print(df_drop_na_initial)

print("\nDataFrame après interpolation:")
print(df_interpolated)

print("\nDataFrame après suppression des NaN (Après Interpolation):")
print(df_drop_na_interpolated)


##exercice11##

import pandas as pd

# Création du DataFrame initial
df_exo11 = pd.DataFrame({
    'A': [1, 4, 7],
    'B': [2, 5, 8],
    'C': [3, 6, 9]
})

# Affichage du DataFrame initial
print("\nDataFrame Initial :")
print(df_exo11)

# Calcul de la somme cumulative
df_cumsum = df_exo11.cumsum()
print("\nDataFrame - Somme Cumulative :")
print(df_cumsum)

# Calcul du produit cumulatif
df_cumprod = df_exo11.cumprod()
print("\nDataFrame - Produit Cumulatif :")
print(df_cumprod)

# Application d'une fonction pour soustraire 1 à chaque élément (correction du warning)
df_subtract_one = df_exo11.map(lambda x: x - 1)
print("\nDataFrame - Soustraction de 1 :")
print(df_subtract_one)



