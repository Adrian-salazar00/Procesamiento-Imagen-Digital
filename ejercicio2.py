import cv2
import matplotlib.pyplot as plt

imagen1 = cv2.imread('rostro.png', cv2.IMREAD_GRAYSCALE)  # Asegúrate de reemplazar 'imagen1.jpg' con tu archivo real
imagen2 = cv2.imread('rostro.png', cv2.IMREAD_GRAYSCALE)  # Asegúrate de reemplazar 'imagen2.jpg' con tu archivo real

if imagen1 is None or imagen2 is None:
    print("Error al cargar las imágenes.")
else:
    print("Imágenes cargadas correctamente.")

orb = cv2.ORB_create()

puntos_clave1, descriptores1 = orb.detectAndCompute(imagen1, None)
puntos_clave2, descriptores2 = orb.detectAndCompute(imagen2, None)


bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
emparejamientos = bf.match(descriptores1, descriptores2)

emparejamientos = sorted(emparejamientos, key=lambda x:x.distance)

imagen_emparejada = cv2.drawMatches(imagen1, puntos_clave1, imagen2, puntos_clave2, emparejamientos[:10], None, flags=2)

plt.figure(figsize=(12, 6))
plt.imshow(imagen_emparejada)
plt.show()