import cv2
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

imagen = cv2.imread('rostro.png')  
convertir_imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2LAB)

if imagen is not None:
    print("Imagen cargada correctamente")
    convertir_imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2LAB)
else:
    print("No se pudo cargar la imagen")

k = 3

convertir_imagen_2d = convertir_imagen.reshape((-1, 3))
convertir_imagen_2d = np.float32(convertir_imagen_2d) 

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
_, labels, centros = cv2.kmeans(convertir_imagen_2d, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

centros2 = np.uint8(centros)
imagen_segmentada = centros2[labels.flatten()]
imagen_segmentada2 = imagen_segmentada.reshape((imagen.shape))

imagen_segmentada_rgb = cv2.cvtColor(imagen_segmentada2, cv2.COLOR_LAB2BGR)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
plt.title('Imagen Original')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(imagen_segmentada_rgb)
plt.title('Imagen Segmentada')
plt.axis('off')

plt.show()