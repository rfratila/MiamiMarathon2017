import pandas as pd
import numpy as np
from sklearn.naive_bayes import BernoulliNB as bnb
from sklearn.utils import shuffle

'''
These three functions prepare the data into dummified form and eliminate any continuous
variables as well as variables like name, or metadata like ID.
'''

# converts hours and minutes to seconds (Niko provided)
def hms2s(hms):
    return reduce(lambda acc, x: acc*60 + x, map(int, hms.split(":")))

# takes in csv, removes columns we don't want (above)

def load_csv(csv_file):
    data = pd.read_csv(csv_file,
                       converters={'Time': hms2s,
                                   'Pace': hms2s, })
    # treats year as category
    data['Year'] = data["Year"].astype('category', ordered=True)
    #get a list of columns
    cols = data.columns.tolist()
    # remove unwanted variables
    list(map(cols.remove, ["Id","Age Category","Time","num","flu","meanTime","sdTime", "day_no"]))
    # return the prepared dataframe
    return data[cols]

# we need dummified data (transforms data into a binary matrix)
def prep_for_nb(data):
    data = pd.get_dummies(data)
    prepared_data = data
    return prepared_data

# returns top 75% of the rows
def get_training_data(data):
    training_data = data[data["Year_2016"] == 0]
    return training_data

# returns vector of what happened in 2016
def get_target_data(data):
    target_data = data["Year_2016"]
    return target_data

def get_testing_data(data):
    testing_data = data[data["Year_2016"] == 1]
    return testing_data

def bnb_sci_kit(training_data, target_data):
    training_data_matrix = training_data.as_matrix()
    target_data_matrix = target_data.as_matrix()
    nb = bnb(alpha=1)
    y_bnb = nb.fit(training_data_matrix, target_data_matrix)
    return y_bnb

def sci_kit_tester(y_bnb, testing_data):
    total = 0
    correct = 0
    year2016 = testing_data['Year_2016']
    tester = testing_data.drop('Year_2016', 1)
    for i in range(len(tester)):
        total += 1
        row = tester.iloc[i, :].as_matrix().reshape(1, -1)
        # x = np.array(row)
        prediction = y_bnb.predict(row)
        actual = year2016.iloc[i]
        if prediction[0] == actual:
            correct += 1
    accuracy = (correct/total)*100
    return accuracy

def bnb_classify(training_data, target_data):
    # calculate prior probabilities
    cat = target_data
    prior_prob = cat.value_counts(normalize=True)
    prior_prob = pd.DataFrame(prior_prob,columns=["prior_probability"])

    likelihoods = {}
    #calculate likelihoods
    for col in list(training_data.columns.values):
        likelihoods[col] = pd.crosstab(training_data["Year_2016"],training_data[col]).apply(lambda r: r/r.sum(), axis=1)

    print likelihoods
    # where I left things
    # need to estimate probabilities of each category and return class label
    # loop through attributes used for categorization and calculate their likelihoods
    # create a dataframe of all of them
    # label the class based on teh highest probability
    # return prediction


#def bnb_tester:
    # read testing data
    # move testing[year 2016] into seperate dataframe
    # run bnb_classify on testing data
    # take returned prediction array (dataframe?) and compare to testing[20160
    # calculate percent correct
    # return accuracy

def main():
    # import data
    all_data = load_csv("full_data.csv")

    # dummify data
    prepared_data = prep_for_nb(all_data)
    # seperate training, target and testing data
    training_data = get_training_data(prepared_data)
    target_data = get_target_data(training_data)
    testing_data = get_testing_data(prepared_data)
    bnb_classify(training_data,target_data)

    # apply sci-kit bnb
    # y_sci_kit_bnb = bnb_sci_kit(training_data.drop("Year_2016", 1),target_data)
    # sci_kit_accuracy = sci_kit_tester(y_sci_kit_bnb, testing_data)

    # apply home grown bnb
    #y_bnb = bnb_classify(training_data.drop("Year_2016",1),target_data)
    #bnb_accuracy = bnb_tester(y_bnb, testing_data)

    #print "scikit accuracy:" + str(sci_kit_accuracy)
    #print "my bnb accuracy:" + str(bnb_accuracy)

main()