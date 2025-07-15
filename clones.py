import pandas as pd

df = pd.read_parquet("data/dados_clones.parquet")

from sklearn import tree

arvore = tree.DecisionTreeClassifier(random_state=42)

df.columns = df.columns.str.strip()

df[['Distância Ombro a ombro', 'Tamanho do crânio', 'Tamanho dos pés']] = df[['Distância Ombro a ombro', 'Tamanho do crânio', 'Tamanho dos pés']].apply(lambda x: x.str.replace('Tipo ', '', regex=False)).astype(int)

y = df['Status']
caracteristicas = ['Massa(em kilos)', 'Estatura(cm)', 'Distância Ombro a ombro', 'Tamanho do crânio', 'Tamanho dos pés', 'Tempo de existência(em meses)']
X = df[caracteristicas]


#print(df.columns)

arvore.fit(X, y)

#print(df[['Massa(em kilos)', 'Estatura(cm)', 'Distância Ombro a ombro', 'Tamanho do crânio', 'Tamanho dos pés', 'Tempo de existência(em meses)']])

#print(df[caracteristicas])

import matplotlib.pyplot as plt

plt.figure(dpi=400, figsize=[4,4])

tree.plot_tree(arvore,
               feature_names=caracteristicas,
               class_names=arvore.classes_,
               max_depth=5,
               filled=True) 

plt.show()

