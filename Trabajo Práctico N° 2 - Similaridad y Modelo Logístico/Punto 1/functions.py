import pandas as pd

def get_data(file_path: str):
    dtype_types = {
    'Region': 'string',
    'Country': 'string',
    'State': 'string',
    'City': 'string',
    'Month': 'int',
    'Day': 'int',
    'Year': 'int',
    'AvgTemperature': 'float'
    }
    data = pd.read_csv(file_path, dtype=dtype_types)
    return data

def filter_city(data, city):
    city_data = data[(data['City'] == city) & (data['AvgTemperature'].notna())]
    city_data = city_data.sort_values(by=['Year', 'Month', 'Day'])
    return city_data

def calculate_daily_variation_with_forward_fd(city_data):
    temperature_variation = []
    city_data = city_data.dropna(subset=['AvgTemperature'])
    temperatures = city_data['AvgTemperature'].values
    for i in range((len(temperatures) - 2)):
            temperature_variation.append(temperatures[i+1] - temperatures[i]) # Diferencia finita hacia adelante
    return temperature_variation

def calculate_daily_variation_with_backward_fd(city_data):
    temperature_variation = []
    city_data = city_data.dropna(subset=['AvgTemperature'])
    temperatures = city_data['AvgTemperature'].values
    for i in range(1, (len(temperatures) - 1)):
        temperature_variation.append(temperatures[i] - temperatures[i-1]) # Diferencia finita hacia atrás
    return temperature_variation

def calculate_daily_variation_with_centered_fd(city_data):
    temperature_variation = []
    city_data = city_data.dropna(subset=['AvgTemperature'])
    temperatures = city_data['AvgTemperature'].values
    for i in range(1, (len(temperatures) - 2)):
        temperature_variation.append((temperatures[i+1] - temperatures[i-1]) / 2) # Diferencia finita centrada
    return temperature_variation

def comparate_variations(city1_data, city2_data):
    comparated_data = [abs(a - b) for a, b in zip(city1_data, city2_data)]
    for variation in comparated_data:
        if variation > 40:
            comparated_data.remove(variation)
    return comparated_data

def filter_by_years(city_data, year1, year2):
    city_year1_data = city_data[city_data['Year'] == year1]
    city_year2_data = city_data[city_data['Year'] == year2]
    return city_year1_data, city_year2_data

def temperatures_in_a_year(city_data_in_year):
    city_data_in_year = city_data_in_year['AvgTemperature']
    temperature_variation = city_data_in_year.diff()
    temperature_variation = temperature_variation.dropna()
    return temperature_variation