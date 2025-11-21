import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn import cluster
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib as mpl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from IPython.display import display
from collections import Counter
import random

# Caricamento del dataset
project = pd.read_csv('WineQT.csv')

print(project)
print(project.head())
# Serie di indici statistici
print(project.describe())
#Ultime righe
print(project.tail())
# Vedi colonne
print(project.columns.tolist())

# Assegno le variabili (X = features, y = target)
X = project[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide',
        'density', 'pH', 'sulphates', 'alcohol', 'Id']]

y = project['quality']

# Verifico il tipo di dati X
print(X.dtypes) 
# Verifico il tipo di dati y
print(y.dtypes)

duplicati = project.duplicated().sum()     # Combinazione funzioni panda. duplicated dà T o F 
                                           # a valori duplicati o meno. sum li somma
print(duplicati)


#I missing values sono valori mancanti che sono indicati con NA o NaN.

mancanti = project.isnull().sum()           # Combinazione funzioni panda. Isnull dà valori bool. 
                                                                  # sum somma in forma numerica
print(mancanti) 

# Creazione di una nuova colonna con classi di alcol
project['Alcohol Category'] = pd.cut(project['alcohol'],
                                     bins=[0, 10, 11.5, 15], 
                                     labels=['Basso', 'Medio', 'Alto'])

# Barplot della qualità media in base alla categoria di alcol

plt.figure(figsize=(8, 6))
sns.barplot(x='Alcohol Category', y='quality', data=project, palette='coolwarm')

plt.title('Qualità media del vino per fascia di alcol')
plt.xlabel('Categoria di Alcol')
plt.ylabel('Qualità media')
plt.show()

# Features = tutte le colonne tranne la target
features = project.drop(["quality", "Id"], axis=1)

# Rimuovi anche eventuali colonne non numeriche o inutili (es: ID o stringhe)
features = features.select_dtypes(include=['int64', 'float64'])  # Tiene solo variabili numeriche

# Target = colonna da prevedere
target = project['quality']

# Standardizzazione
sc = StandardScaler()
scaled_data = sc.fit_transform(features)

# Nuovo DataFrame standardizzato
data_std = pd.DataFrame(scaled_data, columns=features.columns)

# Seleziono alcune variabili numeriche di interesse

data_std['quality'] = target 

cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide',
        'density', 'pH', 'sulphates', 'alcohol']

# Raggruppo per 'quality' sui dati standardizzati e calcolo la media
grouped_std = data_std.groupby('quality')[cols].mean()

# Grafico
grouped_std.plot(kind='bar', figsize=(10, 6), colormap='Set2')
plt.title('Valori medi (standardizzati) per ciascun livello di qualità del vino')
plt.xlabel('Qualità del vino')
plt.ylabel('Valore standardizzato medio')
plt.xticks(rotation=0)
plt.legend(title='Variabile')
plt.tight_layout()
plt.show()

# Calcolo della matrice di correlazione con tutte le colonne disponibili escludendo 'Id'
corr = data_std.corr()

# Visualizzo
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True,
            cbar_kws={"shrink": 0.8}, linewidths=0.5)

plt.title('Matrice di correlazione (dati standardizzati)', fontsize=14)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Semplificazione in classi
data_std['quality_simplified'] = pd.cut(data_std['quality'],
                                  bins=[2, 5, 6, 9],
                                  labels=['Bassa', 'Media', 'Alta'])

# Feature e target
X = data_std.drop(columns=['quality', 'quality_simplified'])
y = data_std['quality_simplified'].astype(str)

# Split train/test (senza ri-scalare)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Decision Tree
clf = DecisionTreeClassifier(criterion='gini', max_depth=3, min_samples_split=4, min_samples_leaf=3, random_state=0)
clf.fit(X_train, y_train)

# Visualizzazione albero
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=X.columns, class_names=['Bassa', 'Media', 'Alta'], filled=True, rounded=True)
plt.title("Decision Tree (con dati standardizzati)")
plt.show()

# Valutazione
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nReport di classificazione:\n", classification_report(y_test, y_pred, target_names=clf.classes_))

#confusion matrix su decision tree

confusion_matrix(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=clf.classes_)
disp.plot()
plt.show()
# --------------------------
# Modelli da confrontare
# --------------------------

models = {
    "Bagging": BaggingClassifier(DecisionTreeClassifier(max_depth=6), n_estimators=200, random_state=42),
    "AdaBoost": AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=200, random_state=42)
}

trainAcc = []
testAcc = []

# --------------------------
# Allenamento e valutazione
# --------------------------
for name, model in models.items():
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    trainAcc.append(accuracy_score(y_train, y_train_pred))
    testAcc.append(accuracy_score(y_test, y_test_pred))
    print(f"\n=== {name} ===")
    print(f"Train Accuracy: {accuracy_score(y_train, y_train_pred):.3f}")
    print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.3f}")
    print("Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=['Bassa', 'Media', 'Alta']))

# --------------------------
# Grafico a barre
# --------------------------
methods = list(models.keys())
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))

ax1.bar(np.arange(len(methods)), trainAcc, color='skyblue')
ax1.set_title("Train Accuracy")
ax1.set_xticks(np.arange(len(methods)))
ax1.set_xticklabels(methods)
ax1.set_ylim(0.4, 1)

ax2.bar(np.arange(len(methods)), testAcc, color='lightgreen')
ax2.set_title("Test Accuracy")
ax2.set_xticks(np.arange(len(methods)))
ax2.set_xticklabels(methods)
ax2.set_ylim(0.4, 1)

plt.suptitle("Confronto tra Ensemble Methods sul dataset del vino")
plt.show()

#Cross validation su decision tree

cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
print("Cross-validation scores:", cv_scores)
print("Media accuratezza cross-validation:", cv_scores.mean())

# Assicura che le etichette siano stringhe
y_train = y_train.astype(str)
y_test = y_test.astype(str)

# ===== KNN =====
k = 12
knn = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=1)
knn.fit(X_train, y_train)

y_pred_knn_train = knn.predict(X_train)
y_pred_knn_test = knn.predict(X_test)

print("=== KNN ===")
print(f"Train Accuracy: {accuracy_score(y_train, y_pred_knn_train)}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_knn_test)}")
print("Report di classificazione:\n", classification_report(y_test, y_pred_knn_test, target_names=knn.classes_))

    # ===== Decision Tree =====


y_pred_tree_train = clf.predict(X_train)
y_pred_tree_test = clf.predict(X_test)

print("\n=== Decision Tree ===")
print(f"Train Accuracy: {accuracy_score(y_train, y_pred_tree_train)}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_tree_test)}")
print("Report di classificazione:\n", classification_report(y_test, y_pred_tree_test, target_names=knn.classes_))

# Creazione del RepeatedStratifiedKFold
# n_ splits numero di fold,n-repeats numero di volte che le cross viene ripetuta
cross_validation = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=1)

# risultati dell'esecuzione
scores = cross_val_score(knn, X, y, cv=cross_validation, scoring='accuracy')

# Accuracy = percentuale di previsioni corrette del modello
print(scores)                                # stampa l'accuracy di tutte le 25 cv
print(f"Mean Accuracy: {scores.mean():.2f}") #stampa media delle 25 cv

#applichiamo la confusion matrix sul knn
confusion_matrix(y_test, y_pred_knn_test)

cm2 = confusion_matrix(y_test, y_pred_knn_test, labels=knn.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=knn.classes_)
disp.plot()
plt.show()

# Definizione profondità da testare
maxdepths = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40]

# Array per salvare le accuratezze
trainAcc = np.zeros(len(maxdepths))
testAcc = np.zeros(len(maxdepths))

# Calcolo accuracies per ciascuna profondità
for i, depth in enumerate(maxdepths):
    clf = DecisionTreeClassifier(max_depth=depth, random_state=0)
    clf.fit(X_train,y_train)
    y_pred_tree_train = clf.predict(X_train)
    y_pred_tree_test = clf.predict(X_test)

    trainAcc[i] = accuracy_score(y_train, y_pred_tree_train)
    testAcc[i] = accuracy_score(y_test, y_pred_tree_test)
# -----------------------
# Trova il punto ottimale
# -----------------------

# Calcolo della differenza assoluta tra train e test (indicatore di overfitting)
diff = np.abs(trainAcc - testAcc)

# Funzione di compromesso: alta test accuracy + bassa differenza
score = testAcc - diff

# Indice della massima combinazione
best_index = np.argmax(score)
best_depth = maxdepths[best_index]

print(f"✅ Profondità ottimale: {best_depth}")
print(f"   Train Accuracy: {trainAcc[best_index]:.3f}")
print(f"   Test Accuracy : {testAcc[best_index]:.3f}")

# -----------------------
# Plot con evidenziazione
# -----------------------
plt.figure(figsize=(10,6))
plt.plot(maxdepths, trainAcc, 'ro-', label='Training Accuracy')
plt.plot(maxdepths, testAcc, 'bv--', label='Test Accuracy')

# Evidenzia il punto ottimale
plt.axvline(x=best_depth, color='g', linestyle='--', label=f'Miglior profondità = {best_depth}')
plt.scatter(best_depth, testAcc[best_index], c='green', s=100, edgecolors='k')

plt.xlabel('Profondità massima dell\'albero (max_depth)')
plt.ylabel('Accuracy')
plt.title('Overfitting/Underfitting nel Decision Tree')
plt.legend()
plt.grid(True)
plt.show()

# Scelta dei valori di k da testare
k_values = list(range(1, 41))  # Da 1 a 40

# Array per salvare le accuratezze
trainAcc = np.zeros(len(k_values))
testAcc = np.zeros(len(k_values))

# Loop su ogni valore di k
for i, k in enumerate(k_values):
    y_train_knn_pred = knn.predict(X_train)
    y_test_knn_pred = knn.predict(X_test)

    trainAcc[i] = accuracy_score(y_train, y_train_knn_pred)
    testAcc[i] = accuracy_score(y_test, y_test_knn_pred)

# Plot delle accuratezze
plt.figure(figsize=(10,6))
plt.plot(k_values, trainAcc, 'ro-', label='Training Accuracy')
plt.plot(k_values, testAcc, 'bv--', label='Test Accuracy')
plt.xlabel('Numero di vicini (k)')
plt.ylabel('Accuracy')
plt.title('Overfitting/Underfitting nel KNN')
plt.legend()
plt.grid(True)
plt.show()

# Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Predizione
y_pred_nb = gnb.predict(X_test)

# Valutazione
print("=== Gaussian Naive Bayes ===")
print(f"Train Accuracy: {accuracy_score(y_train, gnb.predict(X_train))}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_nb)}")
print("Report di classificazione:\n", classification_report(y_test, y_pred_nb, target_names=gnb.classes_))

# Assicura che le etichette siano stringhe
y_train = y_train.astype(str)
y_test = y_test.astype(str)

# Addestramento SVM con kernel RBF
svm_clf = SVC(C=1.0, kernel='rbf', gamma='scale')
svm_clf.fit(X_train, y_train)

# Predizione
y_pred_svm = svm_clf.predict(X_test)

# Valutazione
print("=== Support Vector Machine (RBF) ===")
print(f"Train Accuracy: {accuracy_score(y_train, svm_clf.predict(X_train))}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_svm)}")
print("Report di classificazione:\n", classification_report(y_test, y_pred_svm, target_names=svm_clf.classes_))

# Creazione del RepeatedStratifiedKFold
# n_ splits numero di fold,n-repeats numero di volte che le cross viene ripetuta
cross_validation = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=1)

# risultati dell'esecuzione
scores = cross_val_score(svm_clf, X, y, cv=cross_validation, scoring='accuracy')

# Accuracy = percentuale di previsioni corrette del modello
print(scores)                                # stampa l'accuracy di tutte le 25 cv
print(f"Mean Accuracy: {scores.mean():.2f}") #stampa media delle 25 cv

# PCA: riduzione da 11 feature a 2 componenti principali
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Ora X_pca è un array n x 2 dove ogni riga rappresenta un vino proiettato in uno spazio 2D.

# Dividiamo i dati per l’addestramento del modello

X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.3, random_state=42)

# Addestriamo il modello SVM con kernel RBF

svm_pca = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_pca.fit(X_train_pca, y_train_pca)

# Valutazione su test set
y_pred_pca = svm_pca.predict(X_test_pca)
print("Accuracy su test (PCA + SVM):", accuracy_score(y_test_pca, y_pred_pca))

# Il kernel RBF trova confini curvi, adatti a dati non linearmente separabili
# Visualizziamo il confine decisionale (2D)

# Codifica numerica delle classi per i colori
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Crea griglia 2D per calcolare la decision boundary
h = .02  # passo della griglia
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Predizione su tutti i punti della griglia
Z = svm_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = le.transform(Z)  # Converti da stringhe a numeri
Z = Z.reshape(xx.shape)

# Plot
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.7)

# Punti reali
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1],
                      c=y_encoded, cmap=plt.cm.coolwarm, edgecolors='k', s=30)

plt.title("Confine decisionale SVM con PCA (2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)

# Legenda automatica
plt.legend(*scatter.legend_elements(), title="Classi", labels=le.classes_)
plt.tight_layout()
plt.show()

# Predizione dell’SVM addestrato su tutto il dataset PCA
y_pred_full = svm_pca.predict(X_pca)  # usa il modello corretto: svm_pca

# Maschera booleana per gli errori
error_mask = y != y_pred_full  # confronta etichette reali con predette

# Codifica numerica per la visualizzazione a colori
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_pred_encoded = label_encoder.transform(y_pred_full)

# Visualizzazione
plt.figure(figsize=(10, 6))

# Punti classificati correttamente
plt.scatter(X_pca[~error_mask, 0], X_pca[~error_mask, 1],
            c=y_encoded[~error_mask], cmap=plt.cm.coolwarm,
            edgecolors='k', s=30, alpha=0.7, label='Corretti')

# Punti classificati erroneamente
plt.scatter(X_pca[error_mask, 0], X_pca[error_mask, 1],
            c=y_encoded[error_mask], cmap=plt.cm.coolwarm,
            marker='x', s=50, label='Errori')

plt.title("Errori di classificazione del modello SVM su PCA 2D")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Binarizzazione delle classi (necessario per curva ROC multiclasse)
y_bin = label_binarize(y, classes=['Bassa', 'Media', 'Alta'])
n_classes = y_bin.shape[1]

# Divisione dei dati
X_train, X_test, y_train_bin, y_test_bin = train_test_split(X, y_bin, test_size=0.3, random_state=42, stratify=y)

# Modello one-vs-rest
ovs = OneVsRestClassifier(RandomForestClassifier(random_state=42))
ovs.fit(X_train, y_train_bin)

# Probabilità predette
y_score = ovs.predict_proba(X_test)

# Calcolo fpr, tpr, roc_auc per ogni classe
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC per ogni classe
plt.figure(figsize=(8, 6))
colors = ['darkorange', 'green', 'blue']
class_names = ['Bassa', 'Media', 'Alta']

for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], lw=2, color=colors[i],
             label=f'Classe {class_names[i]} (AUC = {roc_auc[i]:.2f})')

# Linea diagonale
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC Multiclasse')
plt.legend(loc="lower right")
plt.grid()
plt.show()

# Eseguiamo il KMeans su X (presumibilmente già pulito e normalizzato)
k_means = cluster.KMeans(n_clusters=3, max_iter=300, random_state=1)
k_means.fit(X)
labels = k_means.labels_

# Assegniamo i cluster
X_clustered = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
X_clustered['Cluster ID'] = labels

# Centroidi
centroids = pd.DataFrame(k_means.cluster_centers_, columns=X_clustered.columns[:-1])

# Metodo del gomito per scegliere k
SSE = []
numClusters = [1, 2, 3, 4, 5, 6, 7, 8, 9]
for k in numClusters:
    kmeans_test = cluster.KMeans(n_clusters=k, random_state=1)
    kmeans_test.fit(X)
    SSE.append(kmeans_test.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(numClusters, SSE, marker='o')
plt.xlabel('Numero di Cluster')
plt.ylabel('SSE')
plt.title('Metodo del Gomito per KMeans')
plt.grid(True)
plt.show()

data_std = data_std.drop(["quality_simplified"], axis=1)

# Usiamo data_std già creato e standardizzato

sample_labels = data_std.index.astype(str).tolist()

# Clustering gerarchico - metodo Average Link
Z_average = hierarchy.linkage(X, method='average')

# Dendrogramma
plt.figure(figsize=(12, 8))
hierarchy.dendrogram(Z_average, labels=sample_labels, orientation='right', leaf_font_size=7)
plt.title("Hierarchical Clustering - Average Link")
plt.xlabel("Distanza")
plt.ylabel("Campioni (indice)")
plt.tight_layout()
plt.show()

# Clustering gerarchico
Z_average = hierarchy.linkage(data_std, method='average')
cluster_labels = fcluster(Z_average, t=3, criterion='maxclust')

# Matrice delle distanze e ordinamento
distance_matrix = euclidean_distances(data_std)
sorted_indices = np.argsort(cluster_labels)
sorted_distances = distance_matrix[sorted_indices, :][:, sorted_indices]

# Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(sorted_distances, cmap="viridis", xticklabels=False, yticklabels=False)
plt.title("Validation Using Sorted Similarity Matrix (Hierarchical Clustering - Average Link)")
plt.xlabel("Campioni ordinati per cluster")
plt.ylabel("Campioni ordinati per cluster")
plt.tight_layout()
plt.show()

# Carica il dataset
project = pd.read_csv('WineQT.csv')

# Salva separatamente la colonna 'Id'
wine_ids = project['Id']

# Rimuovi 'Id' prima della standardizzazione
features_to_scale = project.drop(columns=['Id'])

# Applica StandardScaler solo sulle feature numeriche
sc = StandardScaler()
scaled = sc.fit_transform(features_to_scale)

# Ricostruisci un DataFrame con le colonne originali (senza 'Id')
scaled_df = pd.DataFrame(scaled, columns=features_to_scale.columns, index=project.index)

# Riaggiungi la colonna 'Id'
scaled_df['Id'] = wine_ids

# Costruzione degli item profiles (standardizzati)
item_profiles_std = {
    scaled_df.iloc[i]['Id']: scaled[i]
    for i in range(len(scaled_df))
}

# Simulazione delle valutazioni degli utenti
np.random.seed(42)
num_users = 10
wine_ids = scaled_df['Id'].unique()
ratings_data = []

for user_id in range(1, num_users + 1):
    rated_wines = np.random.choice(wine_ids, size=np.random.randint(30, 100), replace=False)
    for wine_id in rated_wines:
        base_quality = scaled_df.loc[scaled_df['Id'] == wine_id, 'quality'].values[0]
        simulated_rating = np.clip(np.random.normal(loc=base_quality, scale=0.5), 3, 8)
        ratings_data.append([user_id, wine_id, round(simulated_rating)])

ratings = pd.DataFrame(ratings_data, columns=["userId", "wineId", "rating"])

# Costruzione degli user profiles (media pesata dei profili dei vini valutati)
user_profiles_std = {}

for u in ratings['userId'].unique():
    dfU = ratings[ratings['userId'] == u]
    user_movies = dfU["wineId"].to_numpy()
    user_rating = dfU["rating"].to_numpy()
    user_item_profiles = np.array([item_profiles_std[i] for i in user_movies])
    user_item_profiles_rated = np.array([uip * ur for uip, ur in zip(user_item_profiles, user_rating)])
    user_profile = user_item_profiles_rated.sum(axis=0) / len(user_item_profiles_rated)
    user_profiles_std[u] = user_profile
user_profiles_std[1]

# Seleziona un utente
u = 1
user_profile = user_profiles_std[u]

# Recupera i vini non ancora valutati dall'utente
rated_wines = set(ratings[ratings['userId'] == u]['wineId'])
all_wines = set(scaled_df['Id'])
unrated_wines = list(all_wines - rated_wines)

# Calcolo similarità coseno tra profilo utente e profilo vino
recommendations = []

for wine_id in unrated_wines:
    wine_profile = item_profiles_std[wine_id]
    similarity = cosine_similarity([user_profile], [wine_profile])[0][0]
    recommendations.append((wine_id, similarity))

# Ordina le raccomandazioni per similarità decrescente
top_recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:5]

# Recupera le informazioni dei vini consigliati
recommended_wines = project[project['Id'].isin([wine[0] for wine in top_recommendations])].copy()
recommended_wines["similarity"] = [sim for _, sim in top_recommendations]

# Visualizza
print("Top 5 vini raccomandati per l'utente", u)
display(recommended_wines)

# Calcolo della similarità coseno tra i vini (escludendo 'Id')
X_scaled = scaled_df.drop("Id", axis=1).values
similarity_matrix = cosine_similarity(X_scaled)

# Creazione del grafo orientato
G = nx.DiGraph()

# Aggiunta dei nodi (Id dei vini)
wine_ids = scaled_df['Id'].tolist()
G.add_nodes_from(wine_ids)

# Aggiunta degli archi per similarità > soglia (senza self-loop)
threshold = 0.9
for i, id_i in enumerate(wine_ids):
    for j, id_j in enumerate(wine_ids):
        if i != j and similarity_matrix[i, j] > threshold:
            G.add_edge(id_i, id_j, weight=similarity_matrix[i, j])

# Calcolo del PageRank (ponderato)
pagerank_scores = nx.pagerank(G, alpha=0.85, weight='weight')

# Estrazione dei Top 5 vini per PageRank
top_pagerank = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:5]
top_vini = project[project['Id'].isin([wine_id for wine_id, _ in top_pagerank])].copy()
top_vini["PageRank"] = [score for _, score in top_pagerank]

# Visualizzazione
print("Top 5 vini secondo PageRank:")
display(top_vini)

# Visualizzazione del grafo dei vini
plt.figure(figsize=(12, 10))

# Layout grafico
pos = nx.spring_layout(G, k=0.15, seed=42)

# Disegno nodi
nx.draw_networkx_nodes(G, pos, node_size=30, node_color="lightblue")

# Disegno archi
nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=10, edge_color="gray", width=0.5)

# Titolo e visualizzazione
plt.title("Grafo vino-vino basato su similarità chimica (cosine > 0.9)")
plt.axis("off")
plt.show()

# Layout
pos = nx.spring_layout(G, k=0.15, seed=42)

# Valori PageRank
pr_values = np.array(list(pagerank_scores.values()))
pr_min, pr_max = pr_values.min(), pr_values.max()

# Disegno nodi (ATTENZIONE: node_color prende i valori originali, non normalizzati)
plt.figure(figsize=(14, 12))
nodes = nx.draw_networkx_nodes(
    G,
    pos,
    node_size=30,
    node_color=pr_values,
    cmap=plt.cm.plasma,
    vmin=pr_min, vmax=pr_max  # fondamentale per far funzionare la colorbar
)

# Disegno archi
nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=6, edge_color="gray", width=0.3)

# Colorbar (usiamo il mappable restituito da draw_networkx_nodes)
plt.colorbar(nodes, label="PageRank (normalizzato)")

plt.title("Grafo vino-vino con nodi colorati per PageRank")
plt.axis("off")
plt.show()

# HITS: hub & authority scores
hubs, authorities = nx.hits(G, max_iter=1000, normalized=True)

# PageRank (già calcolato, ma incluso per completezza)
pagerank_scores = nx.pagerank(G, alpha=0.85, weight='weight')

# Creazione DataFrame riassuntivo con tutte le metriche
centrality_df = pd.DataFrame({
    "Id": list(G.nodes),
    "HITS Hub": [hubs[n] for n in G.nodes],
    "HITS Authority": [authorities[n] for n in G.nodes],
    "PageRank": [pagerank_scores[n] for n in G.nodes]
})

# Visualizzazione
print("Metriche di Link Analysis per ciascun vino:")
display(centrality_df.sort_values(by="PageRank", ascending=False).head(10))

# Scatter plot – PageRank vs HITS Authority
plt.figure(figsize=(8, 6))
sns.scatterplot(data=centrality_df, x="PageRank", y="HITS Authority", s=40, alpha=0.7, color="orange")
plt.title("Confronto: PageRank vs HITS Authority")
plt.xlabel("PageRank")
plt.ylabel("HITS Authority")
plt.grid(True)
plt.tight_layout()
plt.show()
############
####   Rete Neurale   #####
############
device = "cuda" if torch.cuda.is_available() else "cpu"
project = pd.read_csv("WineQT.csv")
X = project.drop("quality", axis=1)
y = project["quality"]

# Rietichettatura delle classi: Bassa, Media, Alta
y_cat = y.copy().to_numpy()

# Codifica delle classi
y_cat[y_cat <= 4] = 0           # Bassa
y_cat[(y_cat == 5) | (y_cat == 6)] = 1   # Media
y_cat[y_cat >= 7] = 2           # Alta

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dei dati
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_cat, test_size=0.3, random_state=42)

# y_train e y_test ora sono già array NumPy, quindi:
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Dataset e DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Definizione del modello MLP
class MLP(nn.Module):
    def __init__(self, input_size, hidden1=32, hidden2=16, output_size=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, output_size)
        )

    def forward(self, x):
        return self.model(x)
#la classe creata in precedenza viene assegnata all'oggetto model
model = MLP(input_size=X_train.shape[1])  # .shape[1] ritorna il numero delle colonne così che input_size abbia quel numero come neuroni di imput


counts = Counter(y_train)
print(counts)  # esempio: Counter({1: 289, 2: 45, 0: 9})
total = sum(counts.values())
weights = [total / counts[i] for i in range(3)]  # 3 classi: 0, 1, 2
class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
torch.manual_seed(42)
random.seed(42)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.SGD(model.parameters(), lr=0.001)

y_logits = model(X_test_tensor.to(device))

y_pred_probs = torch.softmax(y_logits, dim=1)
print(y_logits[:5])
print(y_pred_probs[:5])

loss_train=[]
acc_train=[]
epochs=2000
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        pred_labels = torch.argmax(preds, dim=1)
        correct += (pred_labels == yb).sum().item()
        total += yb.size(0)
    
    epoch_acc = 100 * correct / total
    epoch_loss = running_loss / len(train_loader)
    loss_train.append(epoch_loss)
    acc_train.append(epoch_acc)
    if epoch % 50 == 0:
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2f}%")

# Valutazione
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    predicted_classes = predictions.argmax(dim=1).numpy()
    true_classes = y_test_tensor.numpy()

# Report e matrice di confusione
print(confusion_matrix(true_classes, predicted_classes))
print(classification_report(true_classes, predicted_classes))
print(f"Test Accuracy: {accuracy_score(true_classes, predicted_classes) * 100:.2f}%")

# Loss
plt.subplot(1, 2, 1)
plt.plot(loss_train, label='Train Loss')
plt.title('Loss per epoca')
plt.xlabel('Epoca')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(acc_train, label='Train Accuracy', color='green')
plt.title('Accuracy per epoca')
plt.xlabel('Epoca')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()