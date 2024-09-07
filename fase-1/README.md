# Fase 1 - Preprocesamiento y entrenamiento del modelo predictivo

## **Descripción Notebooks**

Los notebooks se encuentran enumerados en el orden en el que deben ser ejecutados para reproducir los resultados:

1. **EDA:** Exploración inicial del conjunto de datos (columnas, tipos de datos, análisis multivariable, valores atípicos, datos faltantes, correlación)
2. **Preprocesamiento**: Tratamiento a columnas con datos faltantes, eliminación de columnas equivalentes, codificación de variables categóricas. El conjunto de datos con los tratamientos aplicados se exportaron en forma de binarios a las rutas `/preprocessed-data/basic/X-preprocessed.p` y `/preprocessed-data/basic/y-preprocessed.p`.
3. **Balance de clases**: Se aplicó submuestreo para balancear las clases. El conjunto de datos con las clases balanceadas se guardaron en forma de binarios en las rutas `/preprocessed-data/final/X-preprocessed.p` y `/preprocessed-data/final/y-preprocessed.p`.
4. **Entrenamiento de modelos**: Aplicación de clasificadores de tipo Random Forest. Las predicciones de clase generadas por el modelo fueron guardadas como binarios en el directorio `model_results`.

> [!NOTE]
> Para correr el modelo generado, únicamente es necesario el notebook `04_Entrenamiento_de_Modelos.ipynb`, ya que el preprocesamiento de los datos se encuentra preguardado.

## **Ejecutar localmente**

> [!NOTE]
> Este repositorio está pensado para ejecutarse desde Google Colab, abriendo los notebooks puedes ver un botón al inicio "Open in Colab".

### **Instalación de dependencias**

En caso de querer ejecutar este proyecto localmente, debes instalar las dependencias necesarias.
Las dependencias están listadas en el archivo `requirements.txt`. Puedes instalar todas las librerías ejecutando:

```bash
pip install -r requirements.txt
```

> [!TIP]
> Se recomienda usar un entorno virtual de python para instalar los requerimientos:
>
> ```bash
> python -m venv venv
>
> # o source venv/bin/activate en linux
> ./venv/Scripts/activate
> ```

> [!IMPORTANT]  
> Los notebooks descargan los datos procesados directamente desde este repositorio, en caso de querer utilizar los resultados locales, se deben actualizar las URL de descarga en los notebooks `03_Balance_de_clases.ipynb` y `04_Entrenamiento_de_modelos.ipynb`.
