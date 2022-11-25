import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.cluster import KMeans

df = pd.read_csv("iris.csv")
df_iris_sem_target = df.drop(['target'], axis=1)

X = np.array(df_iris_sem_target)

kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
df['k-classes'] = kmeans.labels_

print(df)
sb.pairplot(df, x_vars=['sepal length(cm)', 'sepal width(cm)', 'petal length(cm)', 'petal width(cm)'], y_vars=['sepal length(cm)', 'sepal width(cm)', 'petal length(cm)', 'petal width(cm)'], 
hue='target',  palette='Set1', diag_kind='kde', markers=['o', 's', 'D'])
sb.pairplot(df, x_vars=['sepal length(cm)', 'sepal width(cm)', 'petal length(cm)', 'petal width(cm)'], y_vars=['sepal length(cm)', 'sepal width(cm)', 'petal length(cm)', 'petal width(cm)'], 
hue='k-classes', palette='tab10', diag_kind='kde', markers=['s', 'o', 'D'])

plt.show()