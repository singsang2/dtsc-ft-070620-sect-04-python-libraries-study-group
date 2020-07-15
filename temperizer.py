"""An example library for converting temperatures."""


def convert_f_to_c(temperature_f):
    """Convert Fahrenheit to Celsius."""
    temperature_c = (temperature_f - 32) * (5/9)
    return temperature_c




def convert_c_to_f(temperature_c):
    """Convert Celsius to Fahrenheit.
    (X°C × 9/5) + 32 = __°F
""" 
    ## Add your code here
    return (temperature_c*9/5)+32


## CREATE THE ADDITIONAL FUNCTIONS BELOW


# convert_c_to_k
def convert_c_to_k(temperature_c):
    """Convert Celsius to Kelvin.
   X°C + 273.15 = __K
""" 
    ## Add your code here
    return temperature_c+273.15


# convert_f_to_k
def convert_f_to_k(temperature_f):
    """Convert Fahrenheit to Kelvin.
   (X°F − 32) × 5/9 + 273.15 = _K
   """ 
    ## Add your code here
    return (temperature_f-32) * 5/9 + 273.15


# convert_k_to_c
def convert_k_to_c(temperature_k):
    """Convert Kelvin to Celsius
    XK − 273.15 = _°C
""" 
    ## Add your code here
    return temperature_k - 273.15


# convert_k_to_f
def convert_k_to_f(temperature_k):
    """
    Convert Kelvin to Fahrenheit
    (XK − 273.15) × 9/5 + 32 = _°F
    """
    return (temperature_k - 273.15) * 9/5 + 32


## LEVEL UP

# convert_f_to_all
def convert_f_to_all(temperature_f):
    """
    Convert Fahrenheit to Celsius and Fahenheit
    """
    ## Save C and K variables
    temp_c = convert_f_to_c(temperature_f)
    temp_k = convert_f_to_k(temperature_f)
    
    ## Print Message
    print(f"The Temperature {temperature_f} F is:")
    print(f"\t- {round(temp_c,2)} in C")
    print(f"\t- {round(temp_k,2)} in K")


# convert_f_to_all
def convert_f_to_all_level_up(temp_f):
    """
    Convert Fahrenheit to Celsius and Fahenheit
    """
    import pandas as pd
    ## Save C and K in a dict 
    temp_dict = {'C':convert_f_to_c(temp_f),
                 'K': convert_f_to_k(temp_f)}        
    ## Print using List Comprehension
    # [print(f"\t- {round(val,2)} in {key}")for key,val in temp_dict.items()]
    return pd.DataFrame(temp_dict, index=[f'{temp_f} F =']).round(2)
