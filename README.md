# 🛡️ Detección de Fraude en E-Commerce con LightGBM

Sistema de detección de fraude para transacciones de comercio electrónico utilizando técnicas de Machine Learning avanzadas. Este proyecto implementa un modelo de Gradient Boosting (LightGBM) para identificar transacciones fraudulentas con alta precisión.

## 📋 Tabla de Contenidos

- [Descripción](#-descripción)
- [Características](#-características)
- [Dataset](#-dataset)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Resultados](#-resultados)
- [Tecnologías Utilizadas](#-tecnologías-utilizadas)
- [Referencias](#-referencias)
- [Autor](#-autor)

## 🎯 Descripción

Este proyecto implementa un sistema completo de detección de fraude para transacciones de e-commerce que incluye:

- **Análisis Exploratorio de Datos (EDA)** exhaustivo
- **Preprocesamiento** y limpieza de datos
- **Balanceo de clases** usando SMOTETomek
- **Modelo LightGBM** optimizado para clasificación binaria
- **Evaluación completa** con múltiples métricas y visualizaciones

El modelo es capaz de identificar transacciones fraudulentas con un **ROC-AUC de 0.95** y un **F1-Score de 0.88**, demostrando excelente rendimiento en la detección de fraude.

## ✨ Características

- 🔍 **Análisis Exploratorio Completo (EDA)**
  - Análisis de distribución de variables
  - Matriz de correlaciones
  - Análisis de montos y patrones de fraude
  - Visualizaciones profesionales

- ⚙️ **Preprocesamiento Inteligente**
  - Eliminación de variables constantes y no útiles
  - Codificación de variables categóricas
  - Manejo de valores nulos
  - Normalización de datos

- ⚖️ **Balanceo de Dataset**
  - Uso de SMOTETomek (combinación de SMOTE y Tomek Links)
  - Balanceo de clases para mejorar el rendimiento del modelo

- 🤖 **Modelo LightGBM**
  - Gradient Boosting optimizado
  - Early stopping para prevenir overfitting
  - Análisis de importancia de características
  - Validación cruzada (5-fold)

- 📊 **Evaluación Completa**
  - Métricas: Precision, Recall, F1-Score, ROC-AUC
  - Matriz de confusión
  - Curvas ROC y Precision-Recall
  - Visualizaciones guardadas automáticamente

## 📦 Dataset

Este proyecto utiliza el dataset **UAE E-Commerce Fraud Dataset** de Kaggle:

- **Fuente**: [Kaggle - UAE E-Commerce Fraud Dataset](https://www.kaggle.com/datasets/atharvasoundankar/uae-e-commerce-fraud)
- **Tamaño**: 100,000 transacciones
- **Características**: 36 variables (numéricas y categóricas)
- **Variable objetivo**: `is_fraud` (binaria: 0 = No Fraude, 1 = Fraude)
- **Desbalance**: 91.79% No Fraude, 8.21% Fraude

### Variables Principales

- **Transaccionales**: `amount_aed`, `currency`, `payment_method`, `items_count`
- **Dispositivo**: `device_type`, `browser`, `ip_risk_score`
- **Usuario**: `user_account_age_days`, `user_prev_chargebacks`, `user_is_high_risk`
- **Geográficas**: `shipping_city`, `billing_city`, `bin_country`
- **Flags de Fraude**: `fraud_flag_ip`, `fraud_flag_mismatch`, `fraud_flag_velocity`, etc.

## 🚀 Instalación

### Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Pasos de Instalación

1. **Clonar el repositorio** (o descargar los archivos)
   ```bash
   git clone <url-del-repositorio>
   cd E-Commerce-Fraud
   ```

2. **Crear un entorno virtual** (recomendado)
   ```bash
   python -m venv venv
   
   # En Windows
   venv\Scripts\activate
   
   # En Linux/Mac
   source venv/bin/activate
   ```

3. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

4. **Descargar el dataset**
   - Descarga el dataset desde [Kaggle](https://www.kaggle.com/datasets/atharvasoundankar/uae-e-commerce-fraud)
   - Coloca el archivo `uae_ecom_fraud_100k.csv` en la raíz del proyecto

## 💻 Uso

### Ejecución Básica

Simplemente ejecuta el script principal:

```bash
python fraud_detection_lightgbm.py
```

### Qué Hace el Script

El script ejecuta automáticamente las siguientes etapas:

1. **Carga de Datos**: Lee el archivo CSV del dataset
2. **Análisis Exploratorio (EDA)**: Genera estadísticas y visualizaciones
3. **Preprocesamiento**: Limpia y prepara los datos
4. **Balanceo**: Balancea las clases usando SMOTETomek
5. **División Train/Test**: Separa los datos (80% entrenamiento, 20% prueba)
6. **Entrenamiento**: Entrena el modelo LightGBM
7. **Evaluación**: Calcula métricas y genera visualizaciones
8. **Guardado**: Guarda todos los gráficos en la carpeta `Graficos/`

### Salida del Script

El script genera:

- **7 gráficos** guardados en la carpeta `Graficos/`:
  1. `1_distribucion_fraude.png` - Distribución de la variable objetivo
  2. `2_matriz_correlacion.png` - Matriz de correlaciones
  3. `3_analisis_montos.png` - Análisis de montos por fraude
  4. `4_balanceo_dataset.png` - Comparación antes/después del balanceo
  5. `5_importancia_caracteristicas.png` - Top 15 características más importantes
  6. `6_matriz_confusion.png` - Matriz de confusión del modelo
  7. `7_curvas_evaluacion.png` - Curvas ROC y Precision-Recall

- **Métricas en consola**:
  - Classification Report
  - ROC-AUC Score
  - F1-Score
  - Average Precision
  - Validación cruzada

## 📁 Estructura del Proyecto

```
E-Commerce-Fraud/
│
├── fraud_detection_lightgbm.py    # Script principal
├── requirements.txt                # Dependencias del proyecto
├── README.md                       # Este archivo
├── referencia de dataset kaggle.txt # Referencia del dataset
│
├── Graficos/                       # Carpeta con visualizaciones
│   ├── 1_distribucion_fraude.png
│   ├── 2_matriz_correlacion.png
│   ├── 3_analisis_montos.png
│   ├── 4_balanceo_dataset.png
│   ├── 5_importancia_caracteristicas.png
│   ├── 6_matriz_confusion.png
│   └── 7_curvas_evaluacion.png
│
└── uae_ecom_fraud_100k.csv         # Dataset (no incluido, descargar de Kaggle)
```

## 📊 Resultados

### Métricas del Modelo

El modelo entrenado alcanza los siguientes resultados:

| Métrica | Valor |
|---------|-------|
| **ROC-AUC** | 0.9505 |
| **F1-Score** | 0.8821 |
| **Average Precision** | 0.9599 |
| **Accuracy** | 0.89 |
| **Precision (Fraude)** | 0.91 |
| **Recall (Fraude)** | 0.85 |

### Validación Cruzada

- **ROC-AUC Promedio**: 0.9509
- **Desviación Estándar**: ±0.0032
- **Folds**: 5

### Características Más Importantes

Las 5 características más importantes para la detección de fraude son:

1. `bin_country` - País del banco emisor
2. `ip_risk_score` - Puntuación de riesgo de la IP
3. `card_country_match` - Coincidencia del país de la tarjeta
4. `local_hour` - Hora local de la transacción
5. `browser` - Navegador utilizado

## 🛠️ Tecnologías Utilizadas

- **Python 3.8+** - Lenguaje de programación
- **Pandas** - Manipulación y análisis de datos
- **NumPy** - Operaciones numéricas
- **Matplotlib & Seaborn** - Visualización de datos
- **Scikit-learn** - Machine Learning y preprocesamiento
- **LightGBM** - Modelo de Gradient Boosting
- **Imbalanced-learn** - Técnicas de balanceo de clases

## 📚 Referencias

- **Dataset**: [UAE E-Commerce Fraud Dataset](https://www.kaggle.com/datasets/atharvasoundankar/uae-e-commerce-fraud)
- **LightGBM**: [Documentación oficial](https://lightgbm.readthedocs.io/)
- **SMOTETomek**: [Imbalanced-learn documentation](https://imbalanced-learn.org/stable/references/generated/imblearn.combine.SMOTETomek.html)

## 👤 Autor

**Alonso Martin**

- 📧 Email: [alonsomartin1805@gmail.com](alonsomartin1805@gmail.com)

Este proyecto fue desarrollado como parte de un análisis de detección de fraude en e-commerce.

---

## 📝 Notas Adicionales

- El dataset original debe ser descargado desde Kaggle
- Los gráficos se generan automáticamente en la carpeta `Graficos/`
- El modelo utiliza early stopping para prevenir overfitting
- El balanceo de clases es crucial debido al desbalance inicial (11.18:1)

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto es de código abierto y está disponible bajo la licencia MIT.

---

⭐ Si este proyecto te resultó útil, ¡no olvides darle una estrella!
