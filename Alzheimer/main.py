
import os

def renombrar_carpetas(ruta_proyecto):
    # Obtén la lista de carpetas en la ruta del proyecto
    carpetas = [nombre for nombre in os.listdir(ruta_proyecto) if os.path.isdir(os.path.join(ruta_proyecto, nombre))]

    for carpeta in carpetas:
        # Obtén la lista de archivos dentro de la carpeta
        ruta_carpeta = os.path.join(ruta_proyecto, carpeta)
        archivos = [nombre_archivo for nombre_archivo in os.listdir(ruta_carpeta) if os.path.isfile(os.path.join(ruta_carpeta, nombre_archivo))]

        # Enumera los archivos y renómbralos
        for indice, nombre_archivo in enumerate(archivos, start=0):
            nuevo_nombre = f"{indice}.jpg"
            ruta_antigua = os.path.join(ruta_carpeta, nombre_archivo)
            ruta_nueva = os.path.join(ruta_carpeta, nuevo_nombre)

            # Renombrar el archivo
            os.rename(ruta_antigua, ruta_nueva)

        print(f"Archivos de {carpeta} renombrados.")

if __name__ == "__main__":
    # Ruta del proyecto (cambia esto a la ruta de tu proyecto)
    ruta_proyecto = "C:/Users/carlo/Desktop/proyecto-alzheimer/Alzheimer/Resources/data/val"

    renombrar_carpetas(ruta_proyecto)
