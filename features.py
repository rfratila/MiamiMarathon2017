import pandas as pd
import numpy as np
import math

def load_csv(csv_file):
    data = pd.read_csv(csv_file)
    # treats year as category
    data['Year'] = data["Year"].astype('category', ordered=True)
    return data

def get_number_of_past_marathons_in_last_3_years(data):
    data_2014 = data[data["Year"] == 2014]
    data_2015 = data[data["Year"] == 2015]
    data_2016 = data[data["Year"] == 2016]
    data_new = pd.concat([data_2014,data_2015,data_2016])
    data_new = data_new.sort_values(["Id"])
    counts = data_new["Id"].value_counts()
    dict_counts = counts.to_dict()
    count_fun = lambda id: dict_counts[id] if id in dict_counts.keys() else 0
    data['num_runs_2014_2015_2016'] = list(map(count_fun, data['Id']))

def get_number_of_years_since_last_marathon(data):
    data = data.sort_values(["Year"])
    dict = {}
    for index, rows in data.iterrows():
        id = rows["Id"]
        year = rows["Year"]
        dict[id] = 2016-year
    count_fun = lambda id: dict[id]
    data['number_of_years_since_last_marathon'] = list(map(count_fun, data['Id']))
    for index, rows in data.iterrows():
        agefactor = rows["ageFactor"]
    print data.ageFactor.unique()

'''
def get_age_factor_at_last_marathon(data):
    # find there last marathon
    #convert ranges into numbers
    #
    # create dict with their ID and the number of years
    data = data.sort_values(["Year"])
    dict = {}
    for index, rows in data.iterrows():
        id = rows["Id"]
        year = rows["Year"]
        dict[id] = 2016 - year
    count_fun = lambda id: dict[id]
    data['number_of_years_since_last_marathon'] = list(map(count_fun, data['Id']))
    return age_factor_at_last_marathon
'''


def main():
    all_data = load_csv("full_data.csv")
    #number_of_past_marathons = get_number_of_past_marathons_in_last_3_years(all_data)
    number_of_years_since_last_marathon = get_number_of_years_since_last_marathon(all_data)
    # age_factor_at_last_marathon = get_age_factor_at_last_marathon(all_data)

    #all_data.assign(number_of_past_marathons)
    #all_data.assign(number_of_years_since_last_marathon)
    #all_data.assign(age_factor_at_last_marathon)

#Id,Age Category,Sex,Time,Year,ageFactor,day_no,temp,flu,num,sdTime,meanTime,ran_more_than_once

main()