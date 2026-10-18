import os
# Permitir guardar en carpetas locales desde el script
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def cargar_datos_limpios():
    ruta = 'datos/datos_limp/viviendas_limp.csv'
    df = pd.read_csv(ruta)
    X = df.drop(columns=['MedHouseVal'])
    y = df['MedHouseVal']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def entrenar_y_registrar():
    # Usar la nueva carpeta limpia
    mlflow.set_tracking_uri("file:./mlruns_final")
    mlflow.set_experiment("Actividad5_Regresion_Viviendas")
    
    X_train, X_test, y_train, y_test = cargar_datos_limpios()
    
    modelos = {
        "Ridge": (Ridge(), {'alpha': [0.1, 1.0, 10.0]}),
        "XGBoost": (XGBRegressor(random_state=42), {'n_estimators': [50, 100], 'max_depth': [3, 5]})
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for nombre, (modelo, param_grid) in modelos.items():
        print(f"\n--- Entrenando y optimizando: {nombre} ---")
        with mlflow.start_run(run_name=f"Modelo_{nombre}"):
            grid = GridSearchCV(estimator=modelo, param_grid=param_grid, cv=kf, scoring='neg_mean_squared_error')
            grid.fit(X_train, y_train)
            
            mejor_modelo = grid.best_estimator_
            y_pred = mejor_modelo.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            mlflow.log_params(grid.best_params_)
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("R2", r2)
            
            if nombre == "Ridge":
                mlflow.sklearn.log_model(mejor_modelo, "modelo_ridge")
            else:
                os.makedirs("modelos_temp", exist_ok=True)
                path_modelo = "modelos_temp/xgboost_model.pkl"
                joblib.dump(mejor_modelo, path_modelo)
                mlflow.log_artifact(path_modelo, artifact_path="modelo_xgboost")
            
            print(f"Mejores parámetros: {grid.best_params_}")
            print(f"Métricas -> MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")
            print("\nEjecutar los siguientes codigo uno a uno por consola para visualizar en consola de MLflow")
            print("\n set MLFLOW_ALLOW_FILE_STORE=true")
            print("\n mlflow ui --backend-store-uri file:./mlruns_final")

if __name__ == '__main__':
    entrenar_y_registrar()