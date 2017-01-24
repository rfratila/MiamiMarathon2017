import numpy
import csv
import pandas
import math
from linear_regression import bootstrap
from functools import partial
import time

def sigmoid(weight,data):
	prob = 1/(1 + numpy.exp( -numpy.dot(data,weight) ))

	return numpy.clip(prob,0.00001,0.99999)

def error_function(y,weight,data):
	err = - (numpy.dot(y.transpose(),numpy.log(sigmoid(weight,data))) + numpy.dot((1-y.transpose()),numpy.log((1-sigmoid(weight,data)))))
	#import pudb; pu.db
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
	for i in xrange(2):
		weight = get_new_weight(alpha,y,weight,data)
	return weight

	#return lambda in_data: sigmoid(weight,in_data)

#K-Fold cross validation implementation
def cross_validate(k,alpha,weight,data,y):
	numpy.random.shuffle(data)
	
	chunk = k
	i=0
	avg_error = 0
	collection_weights = []
	start_time = time.time()
	while chunk*i < data.shape[0]: 
		test_data = data[chunk*i:chunk*i + chunk]
		test_y = y[chunk*i:chunk*i + chunk]
		train_data = numpy.concatenate((data[:chunk*i],data[chunk*i + chunk:]),axis=0)
		train_y = numpy.concatenate((y[:chunk*i],y[chunk*i + chunk:]),axis=0)

		trained_weight = train(alpha,weight,train_data,train_y)
		collection_weights.append(trained_weight)
		error = 0
		#print ('Testing on chunk [%d,%d]. Training on the rest...'%(chunk*i,chunk*i + chunk))
		for index,test_case in enumerate(test_data):
			error += error_function(test_y[index],trained_weight,test_case)
			#print error_function(test_y[index],trained_weight,test_case)
		
		error /= test_data.shape[0]
		avg_error += error
		i +=1

	avg_error /= (data.shape[0]/k)   #Divide by the total amount of folds trained on

	print "Average error of",avg_error,"where the alpha is", alpha, "in",time.time()-start_time,"seconds"

	return collection_weights

def main():
	training_reserve = 0.7
	validation_reserve = 0.2
	testing_reserve = 0.1
	alpha = [0.01,0.1,1,10,100]           #learning rate
	my_data = pandas.read_csv('full_data.csv',sep=',')

	#d = numpy.random.random((10000,1))
	#data = numpy.array([my_data['Age Category']]).transpose()
	data = numpy.array([my_data[my_data['Year'] != 2016]['Age Category'],
						my_data[my_data['Year'] != 2016]['Time'],
						my_data[my_data['Year'] != 2016]['day_no']]).transpose()#[:200]
	
	d = numpy.concatenate((data,numpy.ones((data.shape[0],1))),axis=1) #for the intercept weight

	w = numpy.random.random((d.shape[1],1))
	y = numpy.array([my_data[my_data['Year'] != 2016]['ran_more_than_once']]).transpose()#[:200]

	my_train = partial(train,alpha,w)
	
	train(alpha,w,d,y)
	for a in alpha:
		coll = cross_validate(10,a,w,d,y)
		import pudb; pu.db
	#print bootstrap(d,y,functional_error,[my_train],num_samples=10)
	
	
		
if __name__ == "__main__":
    main()