# Importo las funciones de utils
from utils import (
    load_data,
    prepare_data,
    scale_data,
    plot_elbow_method,
    split_data,
    train_kmeans_pipeline,
    generate_labels,
    plot_clusters,
    train_random_forest,
    evaluate_model,
    save_model
)

# Importo las librerias principales
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

# --- 1. Carga y Preparación de Datos ---

# Definir la URL de los datos
url = 'https://breathecode.herokuapp.com/asset/internal-link?id=809&path=housing.csv'

# Cargar datos
df = load_data(url)

# Preparar datos (Seleccionar columnas, eliminar duplicados)
data_copy = prepare_data(df)

# --- 2. Modelo No Supervisado (K-Means) ---

# Escalar datos (solo para el gráfico del codo)
data_scaled = scale_data(data_copy)

# Gráfico del codo para determinar K
plot_elbow_method(data_scaled)
# Nota: El notebook determina K=3 visualmente.

# Dividir datos (los datos sin escalar)
X_train, X_test = split_data(data_copy)

# Entrenar K-Means (K=3)
# (Este pipeline escala X_train internamente)
KMean_pipeline = train_kmeans_pipeline(X_train, k=3)

# Generar etiquetas
(
    X_train_clustered,
    X_test_clustered,
    X_sup_train,
    y_sup_train,
    X_sup_test,
    y_sup_test
) = generate_labels(KMean_pipeline, X_train, X_test)

# Graficar clusters
plot_clusters(X_train_clustered, X_test_clustered)

# --- 3. Modelo Supervisado (Random Forest) ---

# Entrenar modelo supervisado (Random Forest)
# Usamos X_train (original) y y_sup_train (labels generadas)
model_rf = train_random_forest(X_sup_train, y_sup_train)

# Evaluar modelo supervisado
evaluate_model(model_rf, X_sup_test, y_sup_test)

# --- 4. Guardar Modelos ---

# Defino las rutas (asumiendo que existe una carpeta 'models' como en el notebook)
models_dir = '../models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    print(f"Directorio '{models_dir}' creado.")

kmeans_path = os.path.join(models_dir, 'kmeans_pipeline.pkl')
rf_path = os.path.join(models_dir, 'model_random_forest.pkl')

# Guardar el pipeline de K-Means
save_model(KMean_pipeline, kmeans_path)
# Guardar el modelo RandomForest
save_model(model_rf, rf_path)

print("\nProceso completado.")