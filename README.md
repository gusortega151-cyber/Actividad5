# Actividad 5: Gestión de Datos, Entrenamiento y Registro con MLOps

## Descripción del Proyecto
Este proyecto desarrolla un pipeline completo de Machine Learning aplicado a un problema de **regresión** (predicción de valores inmobiliarios basado en el dataset de *California Housing*, enriquecido con variables categóricas y tratamiento de nulos). El objetivo es cumplir con las prácticas de *Data-Centric AI* y MLOps, garantizando trazabilidad, versionamiento de datos, validación cruzada, ajuste de hiperparámetros y registro de experimentos.

## Estructura del Repositorio
```text
Actividad5/
│── datos/
│   ├── datos_ini/          # Datos originales y crudos (viviendas_raw.csv)
│   └── datos_limp/         # Datos limpios y estandarizados (viviendas_limp.csv)
│── fuentes/
│   ├── entrena.ipynb       # Cuaderno de análisis y experimentación
│   ├── datos_prep.py       # Funciones automatizadas de limpieza y codificación
│   └── train.py            # Script principal de entrenamiento, Grid Search y MLflow
│── README.md               # Guía de reproducción y documentación técnica
│── CHANGELOG.md            # Bitácora de control de versiones
└── requirements.txt        # Dependencias y librerías del proyecto

## Tecnologías y Librerías Utilizadas
- **Python 3.10**
- **Pandas y NumPy**: Manipulación y cálculo numérico.
- **Scikit-Learn**: Modelos base (Ridge), métricas de regresión, K-Fold y GridSearchCV.
- **XGBoost**: Modelo de ensamble avanzado basado en árboles.
- **MLflow**: Trazabilidad, registro de parámetros, métricas y artefactos.
- **Git y GitHub**: Control de versiones y repositorio remoto.

## Guía de Reproducción
1. Clonar el repositorio en tu equipo local.
2. Crear y activar un entorno virtual con Python 3.10:
   conda create -n actividad5 python=3.10 -y
   conda activate actividad5
3. Instalar las dependencias del archivo `requirements.txt`:
   pip install -r requirements.txt
4. Ejecutar el script de preparación y limpieza de datos:
   python fuentes/datos_prep.py
5. Ejecutar el script de entrenamiento y optimización de hiperparámetros:
   python fuentes/train.py

## Resultados y Análisis Comparativo
Se evaluaron dos algoritmos mediante Validación Cruzada de 5 particiones (5-Fold CV):
- **Ridge Regression (Baseline):** MAE: 0.5335 | RMSE: 0.7451 | R²: 0.5763
- **XGBoost Regressor:** MAE: 0.3146 | RMSE: 0.4727 | R²: 0.8295

*Conclusión técnica:* XGBoost superó con amplio margen al modelo lineal, explicando más del 82% de la varianza del mercado inmobiliario gracias a su capacidad de capturar relaciones no lineales en los datos tabulares.