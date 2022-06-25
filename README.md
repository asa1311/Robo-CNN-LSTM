# Documentación del sistema de detección de eventos de robo

Contiene la ejecución de los códigos de entrenamiento de los modelos YOLO, la configuración de hiperparámetros y entrenamiento e inferencia del modelo en los notebooks de Google Colab y Kaggle, junto al código fuente de la detección de robo en videos.

## Sección 4.1

Para realizar el entrenamiento en Kaggle es necesario actualizar los siguientes paquetes para que funcione correctamente:

```py
!add-apt-repository ppa:ubuntu-toolchain-r/test -y
!apt-get update
!apt-get upgrade libstdc++6 -y
```
El archivo `data.yaml` contiene las rutas con las ímagenes y etiquetas en formato YOLO de entrenamiento y validación, el número de clases y los nombres de cada uno de ellos. El dataset utilizado se encuentra en  www.kaggle.com/dataset/12db5ad0a14028eb14862fd95cea1cbe1120154c4f1f5d818cdf3bd74420cf0d

```py
data_yaml = dict(
    train = '/kaggle/input/dataset/Dataset/images/train',
    val = '/kaggle/input/dataset/Dataset/images/validation',
    nc = 9,
    names = ['Bicycle','Car','Handgun','Knife','Luggage and bags','Motorcycle','Person','Shotgun','Truck']
)
```
Del comando principal de entrenamiento se configuró `--img`, `--batch-size`, `--epochs` , `--data` y `--weights` (para Tiny-YOLOV4 no se configura porque no hay pesos preentrenados para esta versión de Pytorch, por lo que se realiza el entrenamiento desde cero).

```py
!python train.py --img 320 --batch-size 16 --epochs 75 --data data.yaml --weights yolov5n.pt
```
