"""An example library for converting temperatures."""


def convert_f_to_c(temperature_f):
    """Convert Fahrenheit to Celsius."""
    temperature_c = (temperature_f - 32) * (5/9)
    return temperature_c


def convert_c_to_f(temperature_c):
    """Convert Celsius to Fahrenheit."""
    temperature_f = (9/5) * (temperature_c) + 32
    return temperature_f


## CREATE THE ADDITIONAL FUNCTIONS BELOW


# convert_c_to_k
def convert_c_to_k(temperature_c):
    """Convert Celsius to Kelvin."""
    temperature_k = temperature_c + 273.1
    return temperature_k



# convert_f_to_k
def convert_f_to_k(temperature_f):
    temperature_k = convert_c_to_k(convert_f_to_c(temperature_f))
    return temperature_k


# convert_k_to_c
def convert_k_to_c(temperature_k):
    temperature_c = temperature_k - 273.1
    return temperature_c


# convert_k_to_f
def convert_k_to_f(temperature_k):
    temperature_f = convert_c_to_f(convert_k_to_c(temperature_k))
    return temperature_f


## LEVEL UP
def convert_f_to_all(temperature_f):
    print(f'The temperature {temperature_f} F is:')
    print(f'\t-{round(convert_f_to_c(temperature_f),2)} in C')
    print(f'\t-{round(convert_f_to_k(temperature_f),2)} in K')


# convert_f_to_all
def convert_f_to_all_level_up(temperature_f):
    import pandas as pd
    data = {'Celsius': [round(convert_f_to_c(temperature_f),2)],
            'Kelvin': [round(convert_f_to_k(temperature_f),2)],
            'Ferenheigt': [round(temperature_f,2)]}
    df = pd.DataFrame(data)
    return df
