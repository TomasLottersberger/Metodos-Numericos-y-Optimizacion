from graphs import show_error2,show_error, show_both, show_interpolation, show_function_plot, show_points_plot

def main():
    #show_function_plot()
    #show_points_plot()
    #show_interpolation()
    #show_interpolation(use_chebyshev=True)
    #show_both()
    #show_both(use_chebyshev=True)
    #show_error()
    #show_error(use_chebyshev=True,use_median=False)
    #show_error(use_chebyshev=False, use_median=True)
    #show_error(use_chebyshev=True, use_median=True)
    #show_interpolation(method='linear')
    #show_interpolation(use_chebyshev=True,method='linear')
    show_both(method='linear')
    #show_both(use_chebyshev=True,method='linear')
    #show_error(method='linear')
    show_error(use_chebyshev=False,use_median=False,method='linear')
    #show_error(use_chebyshev=False, use_median=True,method='linear')
    show_error(use_chebyshev=False, use_median=True,method='linear')
    #show_both(method='linear')
    #show_error2(method='cubic',num_points_eval=100)

    
if __name__ == "__main__":

    main()
