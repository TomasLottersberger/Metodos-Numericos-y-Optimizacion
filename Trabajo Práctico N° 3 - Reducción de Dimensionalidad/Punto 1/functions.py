import os
import numpy as np
from PIL import Image

def load_images_as_matrix(directory, image_size=(28, 28)):
    images = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.jpeg') or filename.endswith('.png') or filename.endswith('.jpg'):
            img = Image.open(os.path.join(directory, filename)).convert('L')
            img_resized = img.resize(image_size)
            img_vector = np.array(img_resized).flatten()
            images.append(img_vector)
    data_matrix = np.stack(images, axis=0)
    return data_matrix

def svd_compression(data_matrix, d):
    U, S, Vt = np.linalg.svd(data_matrix, full_matrices=False)
    U_d = U[:, :d] #Mantiene solo las primeras d componentes
    S_d = np.diag(S[:d])
    Vt_d = Vt[:d, :]
    data_matrix_d = np.dot(U_d, np.dot(S_d, Vt_d)) #Reconstrucción aproximada
    return data_matrix_d

def calculate_frobenius_error(original_matrix, reconstructed_matrix):
    relative_error = np.linalg.norm(original_matrix - reconstructed_matrix, 'fro')
    return relative_error / np.linalg.norm(original_matrix, 'fro')