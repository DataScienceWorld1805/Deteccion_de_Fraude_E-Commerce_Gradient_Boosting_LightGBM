# -*- coding: utf-8 -*-
"""
Script de Detección de Fraude E-Commerce usando LightGBM
Incluye: EDA, Preprocesamiento, Balanceo y Modelado
"""

import sys
# Configurar encoding UTF-8 para Windows
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, 
                            roc_auc_score, roc_curve, precision_recall_curve,
                            average_precision_score, f1_score)
from imblearn.combine import SMOTETomek
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Crear carpeta para guardar gráficos
graficos_dir = 'Graficos'
if not os.path.exists(graficos_dir):
    os.makedirs(graficos_dir)
    print(f"[OK] Carpeta '{graficos_dir}' creada")
else:
    print(f"[OK] Carpeta '{graficos_dir}' ya existe")

print("="*80)
print("ANÁLISIS DE DETECCIÓN DE FRAUDE E-COMMERCE CON LIGHTGBM")
print("="*80)

# ============================================================================
# 1. CARGA DE DATOS
# ============================================================================
print("\n[1] Cargando datos...")
df = pd.read_csv('uae_ecom_fraud_100k.csv')
print(f"[OK] Dataset cargado: {df.shape[0]:,} filas x {df.shape[1]} columnas")

# ============================================================================
# 2. ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# ============================================================================
print("\n[2] ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
print("-"*80)

# 2.1 Información básica
print("\n2.1 Información General del Dataset")
print(f"Dimensiones: {df.shape}")
print(f"\nColumnas ({len(df.columns)}):")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col}")

# 2.2 Tipos de datos y valores nulos
print("\n2.2 Tipos de Datos y Valores Nulos")
print("\nTipos de datos:")
print(df.dtypes.value_counts())
print("\nValores nulos por columna:")
null_counts = df.isnull().sum()
null_pct = (df.isnull().sum() / len(df)) * 100
null_info = pd.DataFrame({
    'Valores_Nulos': null_counts,
    'Porcentaje': null_pct
})
null_info = null_info[null_info['Valores_Nulos'] > 0].sort_values('Valores_Nulos', ascending=False)
if len(null_info) > 0:
    print(null_info)
else:
    print("  [OK] No hay valores nulos en el dataset")

# 2.3 Estadísticas descriptivas
print("\n2.3 Estadísticas Descriptivas - Variables Numéricas")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(df[numeric_cols].describe())

# 2.4 Distribución de la variable objetivo
print("\n2.4 Distribución de la Variable Objetivo (is_fraud)")
fraud_dist = df['is_fraud'].value_counts()
fraud_pct = df['is_fraud'].value_counts(normalize=True) * 100
print(f"\nDistribución:")
print(f"  No Fraude (0): {fraud_dist[0]:,} ({fraud_pct[0]:.2f}%)")
print(f"  Fraude (1):    {fraud_dist[1]:,} ({fraud_pct[1]:.2f}%)")
print(f"\nRatio de desbalance: {fraud_dist[0]/fraud_dist[1]:.2f}:1")

# Visualización de la distribución
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
fraud_dist.plot(kind='bar', color=['#3498db', '#e74c3c'])
plt.title('Distribución de Fraude', fontsize=14, fontweight='bold')
plt.xlabel('is_fraud', fontsize=12)
plt.ylabel('Frecuencia', fontsize=12)
plt.xticks([0, 1], ['No Fraude', 'Fraude'], rotation=0)
plt.grid(axis='y', alpha=0.3)

plt.subplot(1, 2, 2)
fraud_pct.plot(kind='pie', autopct='%1.1f%%', colors=['#3498db', '#e74c3c'])
plt.title('Proporción de Fraude', fontsize=14, fontweight='bold')
plt.ylabel('')
plt.tight_layout()
plt.savefig(os.path.join(graficos_dir, '1_distribucion_fraude.png'), dpi=300, bbox_inches='tight')
print(f"  [OK] Gráfico guardado: {graficos_dir}/1_distribucion_fraude.png")
plt.close()

# 2.5 Análisis de variables categóricas
print("\n2.5 Análisis de Variables Categóricas")
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"\nVariables categóricas ({len(categorical_cols)}):")
for col in categorical_cols:
    unique_vals = df[col].nunique()
    print(f"  - {col}: {unique_vals} valores únicos")
    if unique_vals <= 10:
        print(f"    Valores: {df[col].value_counts().head().to_dict()}")

# 2.6 Análisis de correlaciones
print("\n2.6 Matriz de Correlación - Variables Numéricas")
corr_matrix = df[numeric_cols].corr()
fraud_corr = corr_matrix['is_fraud'].sort_values(ascending=False)
print("\nCorrelación con is_fraud (Top 15):")
print(fraud_corr.head(15))

# Visualización de correlaciones
plt.figure(figsize=(14, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Matriz de Correlación - Variables Numéricas', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(graficos_dir, '2_matriz_correlacion.png'), dpi=300, bbox_inches='tight')
print(f"  [OK] Gráfico guardado: {graficos_dir}/2_matriz_correlacion.png")
plt.close()

# 2.7 Análisis de montos
print("\n2.7 Análisis de Montos (amount_aed)")
print(f"  Media: {df['amount_aed'].mean():.2f} AED")
print(f"  Mediana: {df['amount_aed'].median():.2f} AED")
print(f"  Desviación estándar: {df['amount_aed'].std():.2f} AED")
print(f"  Mínimo: {df['amount_aed'].min():.2f} AED")
print(f"  Máximo: {df['amount_aed'].max():.2f} AED")

# Comparación de montos por fraude
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
# Usar seaborn boxplot en lugar del método deprecado de pandas
data_for_boxplot = [df[df['is_fraud']==0]['amount_aed'].values, 
                     df[df['is_fraud']==1]['amount_aed'].values]
plt.boxplot(data_for_boxplot, labels=['No Fraude', 'Fraude'])
plt.title('Distribución de Montos por Fraude', fontsize=12, fontweight='bold')
plt.xlabel('is_fraud')
plt.ylabel('Monto (AED)')
plt.grid(axis='y', alpha=0.3)

plt.subplot(1, 2, 2)
df[df['is_fraud']==0]['amount_aed'].hist(alpha=0.7, label='No Fraude', bins=50, color='#3498db')
df[df['is_fraud']==1]['amount_aed'].hist(alpha=0.7, label='Fraude', bins=50, color='#e74c3c')
plt.xlabel('Monto (AED)')
plt.ylabel('Frecuencia')
plt.title('Distribución de Montos', fontsize=12, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(graficos_dir, '3_analisis_montos.png'), dpi=300, bbox_inches='tight')
print(f"  [OK] Gráfico guardado: {graficos_dir}/3_analisis_montos.png")
plt.close()

# 2.8 Análisis de flags de fraude
print("\n2.8 Análisis de Flags de Fraude")
fraud_flags = [col for col in df.columns if 'fraud_flag' in col]
print(f"\nFlags de fraude encontrados ({len(fraud_flags)}):")
for flag in fraud_flags:
    flag_dist = df[flag].value_counts()
    print(f"  - {flag}:")
    print(f"    0: {flag_dist.get(0, 0):,} ({flag_dist.get(0, 0)/len(df)*100:.2f}%)")
    print(f"    1: {flag_dist.get(1, 0):,} ({flag_dist.get(1, 0)/len(df)*100:.2f}%)")

# ============================================================================
# 3. PREPROCESAMIENTO DE DATOS
# ============================================================================
print("\n[3] PREPROCESAMIENTO DE DATOS")
print("-"*80)

# 3.1 Crear copia del dataset
df_processed = df.copy()

# 3.2 Eliminar columnas no útiles para el modelo
columns_to_drop = ['transaction_id', 'user_id', 'timestamp_utc', 'ip_address', 
                   'data_source', 'is_fraud']  # is_fraud es la variable objetivo
print(f"\n3.1 Eliminando columnas no útiles: {columns_to_drop[:-1]}")

# Guardar la variable objetivo antes de eliminar
y = df_processed['is_fraud'].copy()
df_processed = df_processed.drop(columns=columns_to_drop)

# 3.2 Eliminar variables constantes (que no aportan información)
print("\n3.2 Eliminando variables constantes...")
constant_cols = []
for col in df_processed.columns:
    if df_processed[col].nunique() <= 1:
        constant_cols.append(col)
        print(f"  [OK] Eliminada variable constante: {col} (todos los valores son iguales)")

if constant_cols:
    df_processed = df_processed.drop(columns=constant_cols)
else:
    print("  [OK] No se encontraron variables constantes")

# 3.3 Codificación de variables categóricas
print("\n3.3 Codificando variables categóricas...")
label_encoders = {}
categorical_cols_to_encode = df_processed.select_dtypes(include=['object']).columns.tolist()

for col in categorical_cols_to_encode:
    le = LabelEncoder()
    df_processed[col] = le.fit_transform(df_processed[col].astype(str))
    label_encoders[col] = le
    print(f"  [OK] {col}: {len(le.classes_)} categorías codificadas")

# 3.4 Manejo de valores nulos (si existen)
print("\n3.4 Manejo de valores nulos...")
if df_processed.isnull().sum().sum() > 0:
    # Llenar valores nulos en numéricas con la mediana
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_processed[col].isnull().sum() > 0:
            df_processed[col].fillna(df_processed[col].median(), inplace=True)
            print(f"  [OK] {col}: valores nulos llenados con mediana")
else:
    print("  [OK] No hay valores nulos")

# 3.5 Separar características y objetivo
X = df_processed.copy()
print(f"\n3.5 Datos preprocesados:")
print(f"  Características (X): {X.shape}")
print(f"  Variable objetivo (y): {y.shape}")
print(f"  Distribución de y: {y.value_counts().to_dict()}")

# ============================================================================
# 4. BALANCEO DEL DATASET
# ============================================================================
print("\n[4] BALANCEO DEL DATASET")
print("-"*80)

print(f"\nDistribución ANTES del balanceo:")
print(f"  No Fraude (0): {(y==0).sum():,} ({(y==0).sum()/len(y)*100:.2f}%)")
print(f"  Fraude (1):    {(y==1).sum():,} ({(y==1).sum()/len(y)*100:.2f}%)")

# Usar SMOTETomek (combinación de SMOTE y Tomek Links)
print("\nAplicando SMOTETomek para balancear el dataset...")
smote_tomek = SMOTETomek(random_state=42, n_jobs=-1)
X_balanced, y_balanced = smote_tomek.fit_resample(X, y)

print(f"\nDistribución DESPUÉS del balanceo:")
print(f"  No Fraude (0): {(y_balanced==0).sum():,} ({(y_balanced==0).sum()/len(y_balanced)*100:.2f}%)")
print(f"  Fraude (1):    {(y_balanced==1).sum():,} ({(y_balanced==1).sum()/len(y_balanced)*100:.2f}%)")
print(f"  Total: {len(y_balanced):,} muestras")

# Visualización del balanceo
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
y.value_counts().plot(kind='bar', color=['#3498db', '#e74c3c'])
plt.title('ANTES del Balanceo', fontsize=12, fontweight='bold')
plt.xlabel('is_fraud')
plt.ylabel('Frecuencia')
plt.xticks([0, 1], ['No Fraude', 'Fraude'], rotation=0)
plt.grid(axis='y', alpha=0.3)

plt.subplot(1, 2, 2)
pd.Series(y_balanced).value_counts().plot(kind='bar', color=['#3498db', '#e74c3c'])
plt.title('DESPUÉS del Balanceo', fontsize=12, fontweight='bold')
plt.xlabel('is_fraud')
plt.ylabel('Frecuencia')
plt.xticks([0, 1], ['No Fraude', 'Fraude'], rotation=0)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(graficos_dir, '4_balanceo_dataset.png'), dpi=300, bbox_inches='tight')
print(f"  [OK] Gráfico guardado: {graficos_dir}/4_balanceo_dataset.png")
plt.close()

# ============================================================================
# 5. DIVISIÓN TRAIN/TEST
# ============================================================================
print("\n[5] DIVISIÓN TRAIN/TEST")
print("-"*80)

X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_balanced
)

print(f"  Conjunto de entrenamiento: {X_train.shape[0]:,} muestras")
print(f"  Conjunto de prueba: {X_test.shape[0]:,} muestras")
print(f"\n  Distribución en entrenamiento:")
print(f"    No Fraude: {(y_train==0).sum():,} ({(y_train==0).sum()/len(y_train)*100:.2f}%)")
print(f"    Fraude:    {(y_train==1).sum():,} ({(y_train==1).sum()/len(y_train)*100:.2f}%)")
print(f"\n  Distribución en prueba:")
print(f"    No Fraude: {(y_test==0).sum():,} ({(y_test==0).sum()/len(y_test)*100:.2f}%)")
print(f"    Fraude:    {(y_test==1).sum():,} ({(y_test==1).sum()/len(y_test)*100:.2f}%)")

# ============================================================================
# 6. ENTRENAMIENTO DEL MODELO LIGHTGBM
# ============================================================================
print("\n[6] ENTRENAMIENTO DEL MODELO LIGHTGBM")
print("-"*80)

# 6.1 Configuración de parámetros de LightGBM
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42,
    'force_col_wise': True
}

print("\n6.1 Parámetros del modelo:")
for key, value in params.items():
    print(f"  {key}: {value}")

# 6.2 Crear datasets de LightGBM
print("\n6.2 Creando datasets de LightGBM...")
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# 6.3 Entrenar el modelo
print("\n6.3 Entrenando modelo...")
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, test_data],
    valid_names=['train', 'eval'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=100)
    ]
)

print(f"\n[OK] Modelo entrenado con {model.num_trees()} arboles")

# 6.4 Importancia de características
print("\n6.4 Importancia de Características (Top 15)")
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)

print(feature_importance.head(15))

# Visualización de importancia
plt.figure(figsize=(10, 8))
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['importance'], color='#3498db')
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Importancia (Gain)', fontsize=12)
plt.title('Top 15 Características Más Importantes', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(graficos_dir, '5_importancia_caracteristicas.png'), dpi=300, bbox_inches='tight')
print(f"  [OK] Gráfico guardado: {graficos_dir}/5_importancia_caracteristicas.png")
plt.close()

# ============================================================================
# 7. PREDICCIONES Y EVALUACIÓN
# ============================================================================
print("\n[7] PREDICCIONES Y EVALUACIÓN")
print("-"*80)

# 7.1 Predicciones
y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
y_pred = (y_pred_proba >= 0.5).astype(int)

# 7.2 Métricas de evaluación
print("\n7.1 Métricas de Clasificación:")
print(classification_report(y_test, y_pred, target_names=['No Fraude', 'Fraude']))

# 7.3 Matriz de confusión
print("\n7.2 Matriz de Confusión:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Visualización de matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Fraude', 'Fraude'],
            yticklabels=['No Fraude', 'Fraude'])
plt.title('Matriz de Confusión', fontsize=14, fontweight='bold')
plt.ylabel('Verdadero', fontsize=12)
plt.xlabel('Predicho', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(graficos_dir, '6_matriz_confusion.png'), dpi=300, bbox_inches='tight')
print(f"  [OK] Gráfico guardado: {graficos_dir}/6_matriz_confusion.png")
plt.close()

# 7.4 Métricas adicionales
roc_auc = roc_auc_score(y_test, y_pred_proba)
f1 = f1_score(y_test, y_pred)
avg_precision = average_precision_score(y_test, y_pred_proba)

print(f"\n7.3 Métricas Adicionales:")
print(f"  ROC-AUC: {roc_auc:.4f}")
print(f"  F1-Score: {f1:.4f}")
print(f"  Average Precision: {avg_precision:.4f}")

# 7.5 Curvas ROC y Precision-Recall
print("\n7.4 Generando curvas de evaluación...")

# Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='#3498db', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Clasificador Aleatorio')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos', fontsize=12)
plt.ylabel('Tasa de Verdaderos Positivos', fontsize=12)
plt.title('Curva ROC', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)

# Curva Precision-Recall
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
plt.subplot(1, 2, 2)
plt.plot(recall, precision, color='#e74c3c', lw=2, 
         label=f'Precision-Recall (AP = {avg_precision:.4f})')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Curva Precision-Recall', fontsize=14, fontweight='bold')
plt.legend(loc="lower left")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(graficos_dir, '7_curvas_evaluacion.png'), dpi=300, bbox_inches='tight')
print(f"  [OK] Gráfico guardado: {graficos_dir}/7_curvas_evaluacion.png")
plt.close()

# 7.6 Validación cruzada
print("\n7.5 Validación Cruzada (5-fold)...")
cv_scores = cross_val_score(
    lgb.LGBMClassifier(**params, n_estimators=model.num_trees()),
    X_balanced, y_balanced,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='roc_auc',
    n_jobs=-1
)
print(f"  ROC-AUC scores: {cv_scores}")
print(f"  Media: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# ============================================================================
# 8. RESUMEN FINAL
# ============================================================================
print("\n" + "="*80)
print("RESUMEN FINAL")
print("="*80)
print(f"\n[OK] Dataset procesado: {X.shape[0]:,} -> {X_balanced.shape[0]:,} muestras (balanceado)")
print(f"[OK] Caracteristicas utilizadas: {X.shape[1]}")
print(f"[OK] Modelo entrenado: LightGBM con {model.num_trees()} arboles")
print(f"[OK] ROC-AUC en test: {roc_auc:.4f}")
print(f"[OK] F1-Score en test: {f1:.4f}")
print(f"[OK] ROC-AUC promedio (CV): {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
print(f"\n[OK] Gráficos guardados en la carpeta '{graficos_dir}':")
print(f"  1. {graficos_dir}/1_distribucion_fraude.png")
print(f"  2. {graficos_dir}/2_matriz_correlacion.png")
print(f"  3. {graficos_dir}/3_analisis_montos.png")
print(f"  4. {graficos_dir}/4_balanceo_dataset.png")
print(f"  5. {graficos_dir}/5_importancia_caracteristicas.png")
print(f"  6. {graficos_dir}/6_matriz_confusion.png")
print(f"  7. {graficos_dir}/7_curvas_evaluacion.png")
print("\n" + "="*80)
print("ANÁLISIS COMPLETADO EXITOSAMENTE")
print("="*80)
