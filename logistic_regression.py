import numpy
import csv
import pandas
import math
from linear_regression import bootstrap
from functools import partial

def sigmoid(weight,data):
	prob = 1/(1 + numpy.exp( -numpy.dot(data,weight) ))
	return prob

def error_function(y,weight,data):
	err = - (numpy.dot(y.transpose(),numpy.log(sigmoid(weight,data))) + numpy.dot((1-y.transpose()),numpy.log(numpy.clip((1-sigmoid(weight,data)),0.00001,0.99999) )))
	return err

def functional_error(y_hat,y):
	err = - (numpy.dot(y.transpose(),numpy.log(y_hat)) + numpy.dot((1-y.transpose()),numpy.log(numpy.clip((1-y_hat),0.00001,0.99999) )))
	return err

def deriv_error(y,weight,data):

	result = numpy.dot( data.transpose(),(y - sigmoid(weight,data)) )
	return result

def get_new_weight(alpha,y,weight,data):
	new_w = weight + alpha * deriv_error(y,weight,data)
	return new_w

def train(alpha,weight,data,y):
	for i in xrange(10):
		weight = get_new_weight(alpha,y,weight,data)

	return lambda in_data: sigmoid(weight,in_data)

def main():
	training_reserve = 0.7
	validation_reserve = 0.2
	testing_reserve = 0.1
	alpha = 0.1             #learning rate
	my_data = pandas.read_csv('no2013noPrivate.csv',sep=',')

	#d = numpy.random.random((10000,1))
	data = numpy.array([my_data['Age Category']]).transpose()

	d = numpy.concatenate((data,numpy.ones((data.shape))),axis=1) #for the intercept weight

	w = numpy.ones((2,1))
	y = numpy.ones(data.shape)

	my_train = partial(train,alpha,w)
	print bootstrap(d,y,functional_error,[my_train],num_samples=10)
	
	
		
if __name__ == "__main__":
    main()