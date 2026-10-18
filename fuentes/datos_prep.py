import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
import os

def obtener_y_guardar_datos_raw():
    "Descarga el dataset de viviendas y guarda el archivo inicial en datos/datos_ini/."""
    print("Descargando dataset original")
    california = fetch_california_housing(as_frame=True)
    df = california.frame
    
    # Agregamos una variable categorica sintetica
    np.random.seed(42)
    categorias = ['<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'NEAR BAY']
    df['ProximidadOceano'] = np.random.choice(categorias, size=len(df))
    
    # Inyectamos intencionalmente algunos valores nulos y duplicados para demostrar el proceso de limpieza
    df.loc[10:25, 'MedInc'] = np.nan
    df.loc[100:110, 'AveRooms'] = np.nan
    # Duplicamos algunas filas con la misma intencion
    df = pd.concat([df, df.iloc[:50]], ignore_index=True)
    
    os.makedirs('datos/datos_ini', exist_ok=True)
    ruta_raw = 'datos/datos_ini/viviendas_raw.csv'
    df.to_csv(ruta_raw, index=False)
    print(f"Dataset raw guardado exitosamente en: {ruta_raw} con {len(df)} registros.")

def limpiar_datos():
    "Limpieza, tratamiento de nulos, duplicados y codificación."
    print("Proceso de limpieza y estandarizacion")
    ruta_raw = 'datos/datos_ini/viviendas_raw.csv'
    if not os.path.exists(ruta_raw):
        obtener_y_guardar_datos_raw()
        
    df = pd.read_csv(ruta_raw)
    
    # 1 Eliminamos duplicados
    filas_iniciales = len(df)
    df = df.drop_duplicates()
    print(f"Duplicados eliminados: {filas_iniciales - len(df)}")
    
    # 2 Tratamiento de valores nulos
    df['MedInc'] = df['MedInc'].fillna(df['MedInc'].median())
    df['AveRooms'] = df['AveRooms'].fillna(df['AveRooms'].median())
    
    # 3 Codificacion de variables categoricas
    df = pd.get_dummies(df, columns=['ProximidadOceano'], drop_first=True)
    
    # 4 Guardar datos limpios
    os.makedirs('datos/datos_limp', exist_ok=True)
    ruta_limpia = 'datos/datos_limp/viviendas_limp.csv'
    df.to_csv(ruta_limpia, index=False)
    print(f"Dataset limpio guardado exitosamente en: {ruta_limpia}")
    return df

if __name__ == '__main__':
    limpiar_datos()