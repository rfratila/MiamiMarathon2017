
import math
import operator
import numpy as np
import pandas as pd

from functools import reduce
from itertools import product

def bootstrap(x, y, loss_fun, models, num_samples=200, binary_outcome=True):
    """
    Input:
        x - 2D numpy array with samples as rows and features as columns.
        y - 2d column array with value (i) corresponding to the
            sample (i) in x.
        loss_fun - function taking two vertical numpy arrays and returning
            a float loss.
        models - list of model fitting procedures taking similarly shaped
            x, y, and loss function. Each procedure returns a function
            'fit': (1, #features) numpy array -> float
        num_samples - number of bootstrap samples
        binary_outcome - True if y is binary. Extremely significantly peeds up
            the bootstrap error calculation.

    Output:
        err - numpy array with a ".632+ bootstrap error" as described by Efron,
            Tibshirani 1997
    """

    def boot_error(fit_model):
        """
        Finds the bootstrap error based on the given data and the model
        fitting procedure.

        Input:
            fit_model - function that takes 2d array x, 2d column array y,
                and loss_fun and returns a function that takes 2d array x
                and returns 2d column array of predictions y

        Output:
            bootstrap_error - float ".632+ bootstrap error"
        """

        def one_sample(n):
            """
            Helper function for error. Samples from the data, fits a model,
            and finds the loss on the outsample.

            Output:
                in_test_set - a 1D binary array with 1s indicating the presence
                    of observation (i) in the test set
                loss - a 1D array of the loss for observation (i), equal to 0 if
                    (i) was not in the test set.
            """
            in_test_set = np.zeros(n)
            loss = np.zeros(n)

            fit = 0
            while not fit:
                try:
                    train = np.random.choice(n, n)
                    test = np.setdiff1d(np.arange(n), train)

                    x_train, y_train = x[train], y[train]
                    x_test, y_test = x[test], y[test]

                    fit = fit_model(x_train, y_train)
                except numpy.linalg.linalg.LinAlgError as e:
                    pass

            in_test_set[test] = 1
            y_hat = fit(x_test)
            loss[test] = loss_fun(y_test, y_hat)
            print("    done sample")
            return np.concatenate([in_test_set, loss])

        n = len(x)
        q = math.pow(1 - 1/n, n)
        p = 1 - q
        
        fit = fit_model(x, y)
        y_hat = fit(x)

        all_samples = reduce(operator.add, map(one_sample, [n]*num_samples))
        in_test_set, loss = np.split(all_samples, 2)

        if any(in_test_set == 0):
            good_indices = list(filter(lambda i: in_test_set[i] != 0, range(n)))
            loss = loss[good_indices]
            in_test_set = in_test_set[good_indices]

        err_1 = np.mean(loss/in_test_set)
        err_bar = np.mean(list(map(loss_fun, y, y_hat)))
        err_632 = q * err_bar + p * err_1


        if binary_outcome:
            p1 = sum(map(lambda t: t == 1, y))/n
            q1 = sum(map(lambda t: t == 1, y_hat))/n
            gamma = p1 * (1 - q1) + q1 * (1 - p1)
        else:
            unary_loss = lambda a: loss_fun(*a)
            gamma = reduce(operator.add, map(unary_loss, product(y, y_hat)))/n/n
        r = (err_1 - err_bar)/(gamma - err_bar) if err_1 > err_bar and gamma > err_bar else 0

        print("done model")
        return err_632 + (min(err_1, gamma) - err_bar) * (p * q * r) / (1 - q * r)

    return np.array(list(map(boot_error, models)))

def fit_ols(x, y):
    beta = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(x), x)), np.transpose(x)), y)

    return lambda x: np.dot(x, beta)

def l2_norm(y_hat, y):
    return (y_hat-y)**2

def hms2s(hms):
    return reduce(lambda acc, x: acc*60 + x, map(int, hms.split(":")))

def main():
    data = pd.read_csv("Project1_data.csv",
                       converters={'Time': hms2s,
                                   'Pace': hms2s,})
    data['Year'] = data["Year"].astype('category', ordered=True)
    data = data.assign(num=data['Id'].map(dict(data['Id'].value_counts())).astype('float'))
    cols = data.columns.tolist()
    list(map(cols.remove, ['Name', 'Time', 'Pace']))

    x = pd.get_dummies(data[cols]).as_matrix()
    y = data['Time'].as_matrix()

    print(bootstrap(x, y, l2_norm, [fit_ols], 10, False))

if __name__ == "__main__":
    main()
