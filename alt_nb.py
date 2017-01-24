import pandas as pd
import numpy as np
import prettyprint as pp
from sklearn.naive_bayes import BernoulliNB

def hms2s(hms):
    return reduce(lambda acc, x: acc*60 + x, map(int, hms.split(":")))

def load_csv(csv_file):
    data = pd.read_csv(csv_file,
                       converters={'Time': hms2s,
                                   'Pace': hms2s, })
    data['Year'] = data["Year"].astype('category', ordered=True)
    data = data.assign(num=data['Id'].map(dict(data['Id'].value_counts())).astype('float'))
    cols = data.columns.tolist()
    list(map(cols.remove, ['Name', 'Time', 'Pace']))
    data['Id'] = 1
    return data

def prep_for_nb(data):
    prepared_csv = pd.get_dummies(data[cols])
    test_csv = prepared_csv.iloc[1:, 3:]
    return test_csv


def sk_predict(training_data,testing_data):
    bnb = BernoulliNB()
    y_pred = bnb.fit(training_data[0:3],training_data[4])
    y_pred(testing_data)
    return sk_results

def nb_classify():

    return nb_results


def main():
    all_data = load_csv("Project1_data.csv")
    train = all_data[all_data['Year'] != 2016]
    test = all_data[all_data['Year']]
    training_data = prep_for_nb(train)
    testing_data = prep_for_nb(test)
    nb_results = nb_classify(training_data, testing_data)
    sk_results = sk_predict(training_data, testing_data)

    print "nb results"
    print nb_results
    print "-----"
    print sk_results

main()