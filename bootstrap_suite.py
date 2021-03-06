
import math
import operator
import numpy as np
import pandas as pd
import itertools
import time
import multiprocessing

from functools import reduce, partial
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import norm

def metrics(y_hat,y):
    """
    Calculates accuracy, precision, recall, false positive rate, and
    F1 measure and prints them. 

    Input: 
        y_hat - vector of predictions
        y - vector of truth

    Output:
        None

    Note: metrics() pulled from logistic_regression.py because python 
    wouldn't let me import them. 
    """
    TP = np.sum(np.logical_and(y_hat == 1, y ==1))
    TN = np.sum(np.logical_and(y_hat == 0, y ==0))
    FP = np.sum(np.logical_and(y_hat == 1, y ==0))
    FN = np.sum(np.logical_and(y_hat == 0, y ==1))

    print('TP: {}, FP: {}, TN: {}, FN: {}'.format(TP,FP,TN,FN))

    accuracy = float(TP + TN)/(TP + FP + FN + TN)
    precision = float(TP)/(TP + FP)
    recall = float(TP) / (TP + FN)
    false_positive_rate = float(FP) / (FP + TN)
    f1 = 2 * (precision*recall)/(precision+recall)

    
    print("Accuracy: {:.2f}".format(accuracy))
    print("Precision: {:.2f}".format(precision))
    print("Recall: {:.2f}".format(recall))
    print("False positive rate: {:.2f}".format(false_positive_rate))
    print("F1 measure: {:.2f}".format(f1))

def timeit(f, s):
    """
    Helper function to time a function and print a string before and
    after execution.

    Input:
        f - Function with 0 arguments. The output of this function wil
            be returned from timeit.
        s - String, printed before running f() as "S..." and after
            running f() as "Done s..."

    Output:
        x - output of f()
    """
    big_s = s[0].upper() + s[1:]
    small_s = s[0].lower() + s[1:]

    print("{}...".format(big_s))

    t = time.time()
    x = f()

    print("Done {} in {}s.".format(small_s, time.time() - t))

    return x

def parmap(f, X, nprocs=multiprocessing.cpu_count()):
    """
    Paralellized map. 

    Input:
        f - monoid to be applied to each element of X
        X - iterable with each element in the domain of f

    Output:
        y - equivalent to [f(x) for x in X]

    Note: Code adapted from 
    http://stackoverflow.com/revisions/16071616/9.
    """

    def worker(f, q_in, q_out):
        while True:
            i, x = q_in.get()
            if i is None:
                break
            q_out.put((i, f(x)))


    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()

    worker_args = (f, q_in, q_out)
    new_worker = lambda: multiprocessing.Process(target=worker, 
                                                 args=worker_args)

    proc = [new_worker() for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i, x in sorted(res)]

def bootstrap(x, y, loss_fun, models, num_samples=200, binary_outcome=True, metrics=None):
    """
    Input:
        x - pandas DataFrame with samples as rows and features as columns.
        y - pandas DataFrame column with value (i) corresponding to the
            sample (i) in x.
        loss_fun - function taking two vertical numpy arrays and returning
            a float loss.
        models - list of model fitting procedures taking similarly shaped
            x, y, and loss function. Each procedure returns a function
            'fit': (n, #features) pandas DataFrame -> (n, 1) np array
        num_samples - number of bootstrap samples
        binary_outcome - True if y is binary. Extremely significantly peeds up
            the bootstrap error calculation.

    Output:
        err - list with a ".632+ bootstrap error" as described by
            Efron, Tibshirani 1997
    """

    def boot_error(fit_model):
        """
        Finds the bootstrap error based on the given data and the model
        fitting procedure.

        Input:
            fit_model - function that takes pandas Dataframe x, pandas 
                Dataframe  y, and loss_fun and returns a function that takes
                pandas DataFrame x and returns np array of predictions y

        Output:
            bootstrap_error - float ".632+ bootstrap error"
        """

        def one_sample(n):
            """
            Helper function for error. Samples from the data, fits a model,
            and finds the loss on the outsample.

            Output:
                in_test_set - a 1D binary array with 1s indicating the 
                    presence of observation (i) in the test set.

                loss - a 1D array of the loss for observation (i), equal to 0 
                    if (i) was not in the test set.
            """
            in_test_set = np.zeros(n)
            loss = np.zeros(n)

            fit = 0
            while not fit:
                try:
                    train = np.random.choice(n, n)
                    test = np.setdiff1d(np.arange(n), train)

                    fit = fit_model(x[train], y[train])
                except np.linalg.linalg.LinAlgError as e:
                    print("lin_alg_error")
                    pass

            in_test_set[test] = 1
            y_hat = fit(x[test])

            loss[test] = loss_fun(y_hat, y[test])

            return np.concatenate([in_test_set, loss])

        n = len(x)
        q = math.pow(1 - 1/n, n)
        p = 1 - q

        time_sample = lambda x: timeit(lambda:one_sample(n), "Running sample")

        all_samples = reduce(operator.add, 
                             map(time_sample, range(num_samples)))
        in_test_set, loss = np.split(all_samples, 2)

        fit = timeit(lambda:fit_model(x, y), "fitting overall model")
        y_hat = fit(x)
        if metrics is not None: metrics(y_hat, y)

        if any(in_test_set == 0):
            i_in_test_set = lambda i: in_test_set[i] != 0
            good_indices = list(filter(i_in_test_set, range(n)))
            loss = loss[good_indices]
            in_test_set = in_test_set[good_indices]

        err_1 = np.mean(loss/in_test_set)
        err_bar = np.mean(loss_fun(y_hat, y))
        err_632 = q * err_bar + p * err_1

        if binary_outcome:
            p1 = sum(map(lambda t: t == 1, y))/n
            q1 = sum(map(lambda t: t == 1, y_hat))/n
            gamma = p1 * (1 - q1) + q1 * (1 - p1)
        else:
            # gamma = err_bar
            unary_loss = lambda a: loss_fun(*a)
            loss_vector = map(unary_loss, itertools.product(y_hat, y))
            gamma = sum(loss_vector)/n/n
        
        if err_1 > err_bar and gamma > err_bar:
            r = (err_1 - err_bar)/(gamma - err_bar)
        else:
            r = 0

        err1_ = min(err_1, gamma)
        print(err_632 + (err1_ - err_bar) * (p * q * r) / (1 - q * r))
        return err_632 + (err1_ - err_bar) * (p * q * r) / (1 - q * r)

    timed_model = lambda model: timeit(lambda: boot_error(model), "model")
    return np.array(parmap(timed_model, models))
    # return list(map(timed_model, models))

def fit_bnb(x, y, cols=None):
    """
    Fits a binary naive Bayes model using the columns of the data
    specified by cols. 
    """
    if cols is None: cols = np.arange(len(x[0]))
    x = x[:,cols]

    p1 = np.mean(y) # probability(y == 1)
    p = [1-p1, p1]

    def prob_c(cls):
        def bernoulli(col):
            rows = np.where(y == cls)[0]
            return (np.sum(x[rows, col]) + 1) / (len(rows) + 2)
        return np.array(list(map(bernoulli, range(len(cols)))))

    cond_probs = np.array(list(map(prob_c, range(2))))
    log_probs = np.transpose(np.log(cond_probs))

    argmax = lambda x: np.argmax(x, 1).reshape(len(x), 1)

    return lambda x: argmax(np.log(p) + np.dot(x[:,cols], log_probs))

def fit_nb(x, y, cols=None):
    """
    Fits a naive Bayes model using the columns of the data
    specified by cols. The appropriate distribution to fit to each 
    column is detected based on the data in each column.
    """
    if cols is None: cols = np.arange(len(x[0]))
    x = x[:,cols]
    m = len(cols)

    p1 = np.mean(y) # probability(y == 1)
    p = [1-p1, p1]

    def log_cond_prob(cls):
        def bernoulli(col):
            rows = np.where(y == cls)[0]
            p_x1 = (np.sum(x[rows, col]) + 1) / (len(rows) + 2)
            return lambda v: p_x1 if v else 1 - p_x1
        def multinomial(col):
            rows = np.where(y == cls)[0]
            vals = np.sort(np.unique(x[:,col]))
            def p_val_given_c(val):
                num = len(np.where(x[rows, col] == val)[0])
                return (num + 1) / (len(rows) + 2)
            p = {val:p_val_given_c(val) for val in vals}
            return lambda v: p[v] if v in p.keys() else 0
        def normal(col):
            rows = np.where(y == cls)[0]
            mean = np.mean(x[rows,col])
            var = np.var(x[rows,col])
            return partial(norm.pdf, loc=mean, scale=var)
        def log_prob_fun(col):
            """
            Chooses between bernoulli, multinomial, and normal column types.
            """
            vals = np.sort(np.unique(x[:,col]))
            k = len(vals)

            if k == 2:
                f = bernoulli
            elif all(vals == np.arange(k)) or all(vals == np.arange(1, k+1)):
                f = multinomial
            else:
                f = normal
            return lambda x: np.log(f(col)(x))
        return np.array(list(map(log_prob_fun, range(len(cols)))))

    log_probs = np.array(list(map(log_cond_prob, range(2))))

    f = lambda i: lambda xi: sum(log_probs[i,j](xi[j]) for j in range(m))

    def predict(x):
        argmax = lambda x: np.argmax(x, 1).reshape(len(x), 1)
        x = x[:,cols]

        cond_prob = lambda i: np.apply_along_axis(f(i), 1, x)
        cond_probs = np.array(list(map(cond_prob, range(2)))).transpose()

        return argmax(np.log(p) + cond_probs)

    return predict

def fit_cols_nb(cols):
    """
    Partial application function to pass columns to fit_nb.
    """
    return partial(fit_nb, cols=cols)

def fit_ols(x, y, lam=0, cols=None):
    """
    Linear regression fitting function using L2 regularization. Only
    columns specified in cols will be used.
    """
    if cols is None: cols = np.arange(len(x[0]))
    x = x[:,cols]
    n = len(cols)

    xtx_1 = np.linalg.inv(np.dot(np.transpose(x), x) + lam*np.identity(n))
    beta = np.dot(np.dot(xtx_1, np.transpose(x)), y)

    return lambda x: np.dot(x[:,cols], beta)

def fit_cols(cols):
    """
    Partial application function to pass columns to fit_ols.
    """
    return partial(fit_ols, cols=cols)

def sq_err(y_hat, y):
    return (y_hat - y)**2

def main():

    # read data and convert columns to binary dummies
    data = pd.read_csv("full_data.csv")
    cols = data.columns.tolist()
    list(map(cols.remove, ["Age Category", "Id", "Year"]))
    x = pd.get_dummies(data[cols])
    cols = x.columns.tolist()

    # extract outcomes 
    y = data[['Time']].as_matrix()
    y_nb = data[['ran_more_than_once']].as_matrix()

    # create interaction terms
    d = 3
    poly = PolynomialFeatures(degree=d, interaction_only=True)
    x = poly.fit_transform(x)

    # find interaction term colum names
    merge = lambda comb: ":".join(comb)
    i_combs = lambda i: map(merge, itertools.combinations(cols, i))
    cols = list(itertools.chain.from_iterable(map(i_combs, range(1,d+1))))
    cols.insert(0, "Intercept")
    col_inds = lambda names: [cols.index(name) for name in names]

    # feature-sets for linear regression
    model_cols = [["Intercept"], 
                  ["Intercept", "day_no"],
                  ["Intercept", "temp"],
                  ["Intercept", "flu"],
                  ["Intercept", "Sex_F", "Sex_M", "Sex_U"],
                  ["Intercept", "sdTime"],
                  ["Intercept", "day_no", "temp", "flu", "day_no:temp", "day_no:flu", "day_no:temp:flu", "temp:flu", "Sex_F", "Sex_M", "Sex_U", "sdTime"],
                  ["Intercept", "day_no", "temp", "flu", "day_no:temp", "day_no:flu", "day_no:temp:flu", "temp:flu", "Sex_F", "Sex_M", "Sex_U", "sdTime", "num_1", "num_2", "num_3", "num_4", "num_5", "num_6", "num_7", "num_>7", "sdTime:num_1", "sdTime:num_2", "sdTime:num_3", "sdTime:num_4", "sdTime:num_5", "sdTime:num_6", "sdTime:num_7", "sdTime:num_>7", "ageFactor_[20,30)", "ageFactor_[30,40)", "ageFactor_[40,50)", "ageFactor_[50,60)", "ageFactor_[60,70)", "ageFactor_[70,80)", "ageFactor_[80,90)", "ageFactor_[90,100]"]
                 ]

    # feature-sets for naive bayes
    model_cols_nb = [["day_no"],
                     ["temp"],
                     ["flu"],
                     ["Sex_F", "Sex_M", "Sex_U"],
                     ["ageFactor_[20,30)", "ageFactor_[30,40)", "ageFactor_[40,50)", "ageFactor_[50,60)", "ageFactor_[60,70)", "ageFactor_[70,80)", "ageFactor_[80,90)"],
                    ]
    model_cols_nb.reverse()

    # create models
    models = list(map(fit_cols_l2, map(col_inds, model_cols)))
    models += list(map(fit_cols, map(col_inds, model_cols)))
    models_nb = list(map(fit_cols_nb, map(col_inds, model_cols_nb)))

    # run bootstrap
    nb_err = bootstrap(x, y_nb, sq_err, models_nb, 10, metrics=metrics)
    lin_err = bootstrap(x, y, sq_err, models, 200, False)

    # print bootstrap error
    print(nb_err)
    print(lin_err)

if __name__ == "__main__":
    main()
