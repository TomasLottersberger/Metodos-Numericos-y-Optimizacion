import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

def reconstructed_images_graph(original_matrix, reconstructed_matrix, image_size=(28, 28), num_images=5):
    fig, axes = plt.subplots(2, num_images, figsize=(10, 4))
    for i in range(num_images):
        ax = axes[0, i]
        ax.imshow(original_matrix[i].reshape(image_size), cmap='gray')
        ax.axis('off')
        ax.set_title("Original")
        
        ax = axes[1, i]
        ax.imshow(reconstructed_matrix[i].reshape(image_size), cmap='gray')
        ax.axis('off')
        ax.set_title("Reconstruída")
    plt.show()

def plot_similarity_matrices(X, dimensions):
    plt.figure(figsize=(12, 8))
    for i, d in enumerate(dimensions):
        svd = TruncatedSVD(n_components=d)
        X_d = svd.fit_transform(X)
        similarity_matrix = cosine_similarity(X_d)
        ax = plt.subplot(1, len(dimensions), i + 1)
        img = ax.imshow(similarity_matrix, cmap='plasma', interpolation='nearest')
        cbar = plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        plt.title(f"Dimensión: {d}")
    plt.tight_layout()
    plt.show()

def plot_reconstruction_error(errors_list):
    dimensions = []
    errors = []
    for error in errors_list:
        dimensions.append(error[0])
        errors.append(error[1])
    plt.figure(figsize=(8, 6))
    plt.plot(dimensions, errors, marker='o', linestyle='-', color='b')
    plt.xlabel("Número de componentes (Dimensión reducida)")
    plt.ylabel("Error de reconstrucción (Norma de Frobenius)")
    plt.title("Error de reconstrucción mediante la Norma de Frobenius")
    plt.show()