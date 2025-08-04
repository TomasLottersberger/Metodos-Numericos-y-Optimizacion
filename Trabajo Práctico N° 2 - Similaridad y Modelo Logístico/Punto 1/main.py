import functions
import graphs

data = functions.get_data('Punto 1\city_temperature.csv')

def comparation_between_finite_differences():
    city1 = "Buenos Aires"
    city1_data = functions.filter_city(data, city1)
    city1_data = city1_data[(city1_data['Year'] == 2019) & (city1_data['Month'] == 10)]
    variation_with_forward = functions.calculate_daily_variation_with_forward_fd(city1_data)
    variation_with_backward = functions.calculate_daily_variation_with_backward_fd(city1_data)
    variation_with_centered = functions.calculate_daily_variation_with_centered_fd(city1_data)
    graphs.finite_differences_comparation_graph(variation_with_forward, variation_with_backward, variation_with_centered)

def two_cities_of_same_hemisphere():
    city1 = "Seoul"
    city2 = "Moscow"
    city1_data = functions.filter_city(data, city1)
    city1_data = city1_data[(city1_data['Year'] >= 2017) & (city1_data['Year'] <= 2019)]
    city2_data = functions.filter_city(data, city2)
    city2_data = city2_data[(city2_data['Year'] >= 2017) & (city2_data['Year'] <= 2019)]
    city1_variation = functions.calculate_daily_variation_with_centered_fd(city1_data)
    city2_variation = functions.calculate_daily_variation_with_centered_fd(city2_data)
    difference_between_two_cities = functions.comparate_variations(city1_variation, city2_variation)
    graphs.two_cities_comparation_graph(city1, city2, difference_between_two_cities)

def two_cities_of_different_hemisphere():
    city1 = "Seoul"
    city2 = "Buenos Aires"
    city1_data = functions.filter_city(data, city1)
    city1_data = city1_data[(city1_data['Year'] >= 2017) & (city1_data['Year'] <= 2019)]
    city2_data = functions.filter_city(data, city2)
    city2_data = city2_data[(city2_data['Year'] >= 2017) & (city2_data['Year'] <= 2019)]
    city1_variation = functions.calculate_daily_variation_with_centered_fd(city1_data)
    city2_variation = functions.calculate_daily_variation_with_centered_fd(city2_data)
    difference_between_two_cities = functions.comparate_variations(city1_variation, city2_variation)
    graphs.two_cities_comparation_graph(city1, city2, difference_between_two_cities)

def differences_at_same_city_different_years():
    city = "Seoul"
    year1, year2 = 1995, 2017
    city_data = functions.filter_city(data, city)
    city_year1_data, city_year2_data = functions.filter_by_years(city_data, year1, year2)
    city_year1_variation = functions.calculate_daily_variation_with_centered_fd(city_year1_data)
    city_year2_variation = functions.calculate_daily_variation_with_centered_fd(city_year2_data)
    difference_between_years = functions.comparate_variations(city_year1_variation, city_year2_variation)
    graphs.two_years_comparation_graph(city, year1, year2, difference_between_years)

def main():
    comparation_between_finite_differences()
    two_cities_of_same_hemisphere()
    two_cities_of_different_hemisphere()
    differences_at_same_city_different_years()

if __name__ == "__main__":
    main()