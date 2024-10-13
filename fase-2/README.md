# Fase 2 - Contenedor de Docker

## **Archivos principales**

- `train.py`: Script utilizado para entrenar el modelo predictivo a partir de un nuevo set de datos en `input/train.csv`
- `predict.py`: Script que realiza la predicción utilizando el modelo entrenado. Este archivo toma un dataframe como entrada desde la carpeta `input` y genera un archivo de salida en `output/predictions.csv`. Este script espera un modelo `model.pkl` para realizar las predicciones. Este modelo puede ser generado con el script anterior.
- `common.py`: Contiene funciones de preprocesamiento utilizadas por los scripts `train.py` y `predict.py`.

## **Docker**

El proyecto incluye un archivo `Dockerfile` que facilita la ejecución del entorno de manera aislada utilizando Docker. Para ejecutar el contenedor:

1. Construir la imagen Docker:

```bash
 docker build -t intro-ia-fase-2 .
```

2. Ejecutar el contenedor
   > [!IMPORTANT]
   > El contenedor Docker acepta una variable de entorno `ACTION`, que puede tomar los valores de `predict` o `train`. En caso de no estar presente, por defecto se realizará una predicción con el modelo `model.pkl` y el CSV `input/test.csv`.

```bash
docker run -e ACTION=predict --rm -v $pwd/output_container:/app/output intro-ia-fase-2
```

> [!NOTE]
> La salida generada por el contenedor (ya sea el nuevo modelo, o las predicciones generadas), serán accesibles en la carpeta `output_container` que se generará al correr el contenedor.

> [!NOTE]
> Es necesario tener Docker instalado en tu PC. Si utilizas un sistema operativo Linux, puedes instalarlo directamente. En cambio, si usas Windows, necesitarás instalar el Subsistema de Windows para Linux (WSL) para poder utilizar Docker.
