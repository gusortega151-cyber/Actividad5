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