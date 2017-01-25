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
    data = data.sort_values(["Id"])
    return data['num_runs_2014_2015_2016']

def get_number_of_years_since_last_marathon(data):
    data = data.sort_values(["Year"])
    dict = {}
    for index, rows in data.iterrows():
        id = rows["Id"]
        year = rows["Year"]
        dict[id] = 2016-year
    count_fun = lambda id: dict[id]
    data['number_of_years_since_last_marathon'] = list(map(count_fun, data['Id']))
    data = data.sort_values(["Id"])
    return data['number_of_years_since_last_marathon']

def get_age_factor_at_last_marathon(data):
    age_dict = {'[40,50)':4, '[30,40)':3, '[20,30)':2, '[60,70)':6, '[50,60)':5, '[70,80)':7, '[10,20)':1,
 '[80,90)':8, '[90,100]':9}

    for index, rows in data.iterrows():
        ageFact = rows["ageFactor"]
        id = rows["Id"]
        replacement_age = age_dict[ageFact]
        data.set_value(index,"ageFactor",replacement_age)
    age_dict_id = {}
    data = data.sort_values(["Year"])
    for index, rows in data.iterrows():
        id = rows["Id"]
        age_factor = rows["ageFactor"]
        age_dict_id[id] = age_factor
    count_fun = lambda id: age_dict_id[id]
    data["age_factor_at_last_marathon"] = list(map(count_fun, data["Id"]))
    data = data.sort_values(["Id"])
    return data["age_factor_at_last_marathon"]

def main():
    all_data = load_csv("full_data.csv")
    all_data = all_data.sort_values(["Id"])
    print len(all_data["Id"].tolist())
    number_of_past_marathons = get_number_of_past_marathons_in_last_3_years(all_data)
    number_of_years_since_last_marathon = get_number_of_years_since_last_marathon(all_data)
    age_factor_at_last_marathon = get_age_factor_at_last_marathon(all_data)
    print len(number_of_past_marathons.index)
    print len(number_of_years_since_last_marathon.index)
    print len(age_factor_at_last_marathon.index)

    all_data["number_of_past_marathons"]=number_of_past_marathons
    all_data["number_of_years_since_last_marathon"]=number_of_years_since_last_marathon
    all_data["age_factor_at_last_marathon"]=age_factor_at_last_marathon
    final_data = all_data
    print final_data[final_data["Id"] == 8]
    print len(final_data.index)
    final_data.to_csv("full_data_1.csv")
main()