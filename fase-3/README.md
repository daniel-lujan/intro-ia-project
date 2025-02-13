# Fase 3 - Aplicación REST

## **Archivos principales**

- `apirest.py`: Aplicación de FastAPI para exponer el endpoint /predict que será utilizado para realizar predicciones con base a un modelo de ML, además expone el endpoint /train que será utilizado para reentrenar el modelo.
- `client.py`: Script diseñado para enviar datos a `/predict` y obtener una predicción, además se llama al endpoint `/train` para solicitar el reentrenamientio del modelo.
- `ml_model/__init__.py`: Contiene la clase `Model` que se encarga de incializar el modelo de ML de forma global y realizar las predicciones.

## **Docker**

El proyecto incluye un archivo `docker-compose.yaml` que define y ejecuta los servicios para exponer y probar la API.

Hay dos Dockerfiles:

- `Dockerfile.api`: Define la imagen de la API REST.
- `Dockerfile.client`: Define la imagen del cliente que se conecta a la API.

Para construir y ejecutar los contenedores, solo es necesario ejecutar el siguiente comando:

```bash
 docker-compose up
```

Esto construye las imágenes de la api y el cliente, luego ejecutará los contenedores. Una vez la API esté funcionando, se lanza el contenedor `client`. Al finalizar, se verán los _logs_ de las llamadas a la API:

```bash
client-1  | Making Predict request
client-1  | {'earnings_more_than_50k': True}
client-1  | Predict request took 0.03s
client-1  | ===================================
client-1  | Making Train request
client-1  | {'message': 'Training complete'}
client-1  | Train request took 9.30s
client-1 exited with code 0
```

> [!NOTE]
> Es necesario tener Docker instalado en tu PC. Si utilizas un sistema operativo Linux, puedes instalarlo directamente. En cambio, si usas Windows, necesitarás instalar el Subsistema de Windows para Linux (WSL) para poder utilizar Docker.
