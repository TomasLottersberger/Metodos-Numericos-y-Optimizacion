import functions, graphs

def main():
    directory_path = 'Punto 1\dataset_imagenes'
    data_matrix = functions.load_images_as_matrix(directory_path)
    image_size = (28, 28)
    d_values = list(range(1, 17))
    error_for_dimension = []
    for d in d_values:
        reconstructed_matrix = functions.svd_compression(data_matrix, d)
        relative_error = functions.calculate_frobenius_error(data_matrix, reconstructed_matrix)
        error_for_dimension.append((d, relative_error))
        print(f"Dimension d={d}, Error de Frobenius relativo: {relative_error:.2f}")
        #if d == 5 or d == 10 or d == 15:
        #    graphs.reconstructed_images_graph(data_matrix, reconstructed_matrix, image_size=image_size)
        if relative_error < 0.10:
            print(f"Con d={d}, el error de compresion es inferior al 10%. Este valor de d es adecuado.")
    #graphs.plot_similarity_matrices(data_matrix, [5, 10, 15])
    #graphs.plot_reconstruction_error(error_for_dimension)

if __name__ == "__main__":
    main()