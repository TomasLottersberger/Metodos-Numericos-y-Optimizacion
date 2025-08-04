import matplotlib.pyplot as plt

def finite_differences_comparation_graph(forward, backward, centered):
    plt.figure(figsize=(12, 6))
    plt.plot(forward, label='Forward')
    plt.plot(backward, label='Backward', linestyle='--')
    plt.plot(centered, label='Centered')
    plt.xlabel('Tiempo (en días)')
    plt.ylabel('Variación (°F/día)')
    plt.title(f'Comparación de la Tasa de Variación de Temperatura medido con diferencias finitas hacia adelante, hacia atrás y centradas')
    plt.legend()
    plt.show()

def two_cities_comparation_graph(city1, city2, differences):
    plt.figure(figsize=(12, 6))
    plt.plot(differences)
    for i in range(0, len(differences), 365):
        plt.axvline(x=i, color='red', linestyle='--', linewidth=0.8)
    plt.xlabel('Tiempo (en días)')
    plt.ylabel('Variación (°F/día)')
    plt.title(f'Comparación de la Tasa de Variación de Temperatura: {city1} vs {city2}')
    plt.show()

def two_years_comparation_graph(city, year1, year2, differences):
    plt.figure(figsize=(12, 6))
    plt.plot(differences)
    plt.xlabel('Tiempo (en días)')
    plt.ylabel('Variación (°F/día)')
    plt.title(f'Comparación de la Tasa de Variación de Temperatura en {city}: {year1} vs {year2}')
    plt.show()