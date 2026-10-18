```markdown
# Changelog - Actividad 5

Todos los cambios de este proyecto serán documentados en este archivo.

## [1.0.0] - 2026-10-18
### Añadido
- Estructura base del proyecto con directorios `datos/datos_ini`, `datos/datos_limp` y `fuentes`.
- Script automatizado `datos_prep.py` para la descarga, inyección controlada de nulos, eliminación de duplicados y codificación One-Hot de variables categóricas.
- Script `train.py` para el entrenamiento con validación cruzada (*K-Fold*), optimización con *Grid Search* para Ridge y XGBoost, y registro de corridas, métricas y artefactos mediante MLflow.
- Configuración de dependencias en `requirements.txt` y exclusiones en `.gitignore`.
- Documentación técnica inicial en `README.md`.