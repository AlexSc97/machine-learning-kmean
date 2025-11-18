import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


# Funciones para cargar datos y preparar datos

def load_data(url):
    """
    :param url: Url con la data .csv
    :return: un dataframe de pandas
    """
    print(f"Cargando datos desde la URL: {url}")
    df = pd.read_csv(url)
    print(f"Datos cargados correctamente")
    return df


# Función de limpieza y preparación de datos

def prepare_data(df):
    """
    :param df: DataFrame original de California Housing
    :return: DataFrame procesado (columnas seleccionadas, duplicados eliminados)
    """
    print("Seleccionando columnas 'Latitude', 'Longitude', 'MedInc'")
    # Selecciono las columnas que me interesan.
    data_copy = df[['Latitude', 'Longitude', 'MedInc']].copy()

    print(f"Buscando duplicados")
    # Busca si hay duplicados y los elimina
    duplicates = data_copy.duplicated().sum()
    if duplicates > 0:
        print(f"Eliminando {duplicates} duplicados")
        data_copy = data_copy.drop_duplicates()
        # Verifico que se hayan eliminado
        print(f"Duplicados restantes: {data_copy.duplicated().sum()}")
    else:
        print("No se encontraron duplicados")

    # Busca si hay nulos
    print("Buscando valores nulos")
    nulls = data_copy.isnull().sum().sum()
    if nulls == 0:
        print("No se encontraron valores nulos")
    else:
        print(f"Se encontraron {nulls} valores nulos (se omitirá la imputación según el notebook)")

    return data_copy


def scale_data(df):
    """
    :param df: DataFrame con datos limpios
    :return: Datos escalados (para el codo)
    """
    print("Escalando datos con StandardScaler para el método del codo")
    # Creo el escalador scaler
    scaler = StandardScaler()
    # Entreno y transformo la data con el escalador
    data_scaled = scaler.fit_transform(df)
    print("Datos escalados correctamente")
    return data_scaled


def plot_elbow_method(data_scaled):
    """
    :param data_scaled: Datos escalados
    :return: Muestra el gráfico del codo
    """
    print("Calculando inercia para el método del codo (K=1 a 10)")
    # Creo una lista donde se almacenaran las k
    k_list = []
    k_range = range(1, 11)
    for k in k_range:
        # Configuro n_init=10 como en el notebook
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(data_scaled)
        k_list.append(km.inertia_)

    print("Graficando el método del codo")
    # Grafico el codo
    sns.set_style('whitegrid')
    plt.figure(figsize=(10, 10))
    sns.lineplot(x=k_range, y=k_list, marker='o', linestyle='--', color='b')
    plt.title('Metodo del codo')
    plt.xlabel('Numero de clusters (K)')
    plt.ylabel('Inercia (suma de los errores cuadrados)')
    plt.xticks(k_range)
    plt.show()


def split_data(df):
    """
    :param df: DataFrame limpio (sin escalar)
    :return: data dividida en entrenamiento y prueba (solo X)
    """
    print("Dividiendo datos en entrenamiento y prueba (80/20)")
    # Divido la data de entrenamiento de las caracteristicas
    X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)
    print("Datos divididos correctamente")
    return X_train, X_test


def train_kmeans_pipeline(X_train, k=3):
    """
    :param X_train: Datos de entrenamiento (sin escalar)
    :param k: Número de clusters (default=3 según el codo)
    :return: Pipeline de K-Means entrenado
    """
    print(f"Creando y entrenando pipeline de K-Means con K={k}")
    # Establezco la variable para crear el flujo
    KMean_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('kmeans', KMeans(n_clusters=k, random_state=42, n_init=10))
    ])
    # Entreno el pipeline solo con los datos de entrenamiento
    KMean_pipeline.fit(X_train)
    print("Pipeline K-Means entrenado correctamente")
    return KMean_pipeline


def generate_labels(pipeline, X_train, X_test):
    """
    :param pipeline: Pipeline K-Means entrenado
    :param X_train: Datos X de entrenamiento
    :param X_test: Datos X de prueba
    :return: X_train_clustered, X_test_clustered, X_sup_train, y_sup_train, X_sup_test, y_sup_test
    """
    print("Generando etiquetas de cluster para train y test")
    # Obtengo las etiquetas para la visualizacion de train
    train_labels = pipeline.predict(X_train)
    X_train_clustered = X_train.copy()
    X_train_clustered['cluster'] = train_labels

    # Obtengo las etiquietas para la visualizacion de test
    test_labels = pipeline.predict(X_test)
    X_test_clustered = X_test.copy()
    X_test_clustered['cluster'] = test_labels

    print("Preparando datos X/y para el modelo supervisado")
    # Preparo datos para modelo supervisado
    X_sup_train = X_train
    y_sup_train = train_labels
    X_sup_test = X_test
    y_sup_test = test_labels

    return X_train_clustered, X_test_clustered, X_sup_train, y_sup_train, X_sup_test, y_sup_test


def plot_clusters(X_train_clustered, X_test_clustered):
    """
    :param X_train_clustered: Datos de entrenamiento con clusters
    :param X_test_clustered: Datos de prueba con clusters
    :return: Muestra el gráfico de dispersión de clusters
    """
    print("Graficando clusters (Train y Test)")
    plt.figure(figsize=(10, 8))
    # Gráfico de entrenamiento (puntos)
    sns.scatterplot(
        data=X_train_clustered,
        x='Longitude',
        y='Latitude',
        hue='cluster',
        palette='viridis',
        s=50,
        alpha=0.5
    )
    # Gráfico de prueba (X)
    sns.scatterplot(
        data=X_test_clustered,
        x='Longitude',
        y='Latitude',
        hue='cluster',
        palette='viridis',
        s=100,
        legend=False,
        marker='X',
        alpha=0.5
    )
    plt.title('Distribución Geográfica de Clusters (Train y Test)')
    plt.show()


def train_random_forest(X_train, y_train):
    """
    :param X_train: Características de entrenamiento (supervisado)
    :param y_train: Etiquetas de entrenamiento (clusters)
    :return: Modelo RandomForest entrenado
    """
    print(f"Entrenando modelo de RandomForestClassifier")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    # Validación cruzada (como en el notebook)
    print("Realizando validación cruzada (CV=5)")
    cv_scores = cross_val_score(model, X_train, y_train,
                                cv=5, scoring='accuracy', n_jobs=-1)

    print(f"Scores de Accuracy por fold: {np.round(cv_scores, 4)}")
    print(f"Accuracy Media (CV): {cv_scores.mean():.4f}")
    print(f"Desviación Estándar (CV): {cv_scores.std():.4f}")

    # Entreno el modelo.
    model.fit(X_train, y_train)
    print("Modelo RandomForest entrenado correctamente")
    return model


def evaluate_model(model, X_test, y_test):
    """
    :param model: Modelo supervisado entrenado
    :param X_test: Características de prueba
    :param y_test: Etiquetas de prueba
    :return: Imprime el reporte de evaluación
    """
    print("Evaluando modelo RandomForest en conjunto de prueba")
    # Realizar predicciones
    y_pred = model.predict(X_test)

    # Evaluar el modelo
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"\n--- Evaluación del Modelo Supervisado (Random Forest) ---")
    print(f"Accuracy (Precisión) en conjunto de prueba: {accuracy:.4f}")
    print("\nReporte de Clasificación:")
    print(report)


def save_model(model, path):
    """
    :param model: modelo (pipeline o clasificador)
    :param path: ruta de guardado (ej. '../models/mi_modelo.pkl')
    :return: Guarda el modelo.pkl en la ruta indicada
    """
    print(f"Guardando modelo en: {path}")
    # Abro el archivo en modo 'write binary' (wb)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print("Modelo guardado correctamente")