import numpy
import csv
import pandas
import math
from linear_regression import bootstrap
from functools import partial
import time
from collections import OrderedDict
import pylab

from linear_regression import sq_err

def sigmoid(weight,data):
	prob = 1/(1 + numpy.exp( -numpy.dot(data,weight) ))

	return numpy.clip(prob,0.00001,0.99999)

def error_function(y,weight,data):
	err = - (numpy.dot(y.transpose(),numpy.log(sigmoid(weight,data))) + numpy.dot((1-y.transpose()),numpy.log((1-sigmoid(weight,data)))))
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

#Given the necessary parameters, it will output the new trained weights
#Uncomment the lines for printing the training graph
def train(alpha,weight,data,y):

	store = OrderedDict(error=[],iteration=[])
	for i in xrange(1000):
		store['error'].append(error_function(y,weight,data)[0][0])
		#store['iteration'].append(i)
		weight = get_new_weight(alpha,y,weight,data)
		#print error_function(y,weight,data)[0][0]
	
	#pylab.subplot(1,1,1)
	#pylab.plot(store['iteration'],store['error'], '-ro',label='Training Error')
	#pylab.xlabel("Iteration")
	#pylab.ylabel("Error")
	#pylab.legend(loc='upper right')
	#pylab.title('Alpha is %.2g'%alpha)
	#pylab.show()
	print 'Most recent training error:',store['error'][-1]
	print 'Standard deviation on latter half of training cycle:', numpy.std(store['error'][len(store['error'])/2:])
	
	
	return weight
	
	#return lambda in_data: sigmoid(weight,in_data) #Used for bootstrap

#K-Fold cross validation implementation
def cross_validate(k,alpha,weight,data,y):
	numpy.random.shuffle(data)
	
	chunk = int(math.ceil(data.shape[0]/k))
	
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
		print ('Testing on chunk [%d,%d]. Training on the rest...'%(chunk*i,chunk*i + chunk))
		error = error_function(test_y,trained_weight,test_data)[0][0]
		y_hat = numpy.round(sigmoid(trained_weight,test_data)); calculate_metrics(y_hat,test_y)

		avg_error += error
		i +=1

	avg_error /= k   #Divide by the total amount of folds trained on

	print("Average testing error of",avg_error,"where the alpha is", alpha, "in",time.time()-start_time,"seconds")

	return collection_weights

#This method is responsible for calculating the classification metrics
def calculate_metrics(y_hat,y):
	TP = numpy.sum(numpy.logical_and(y_hat == 1, y ==1))
	TN = numpy.sum(numpy.logical_and(y_hat == 0, y ==0))
	FP = numpy.sum(numpy.logical_and(y_hat == 1, y ==0))
	FN = numpy.sum(numpy.logical_and(y_hat == 0, y ==1))

	print('TP: {}, FP: {}, TN: {}, FN: {}'.format(TP,FP,TN,FN))
	accuracy = precision = recall = f1 = false_positive_rate = 0.0
	try:
		accuracy = float(TP + TN)/(TP + FP + FN + TN)
		precision = float(TP)/(TP + FP)
		recall = float(TP) / (TP + FN)
		false_positive_rate = float(FP) / (FP + TN)
		f1 = 2 * (precision*recall)/(precision+recall)
	except ZeroDivisionError:
		print('Cannot compute all metrics. Check predictions.')
	
	print("Accuracy: {:.2f}".format(accuracy))
	print("Precision: {:.2f}".format(precision))
	print("Recall: {:.2f}".format(recall))
	print("False positive rate: {:.2f}".format(false_positive_rate))
	print("F1 measure: {:.2f}".format(f1))

def main():
	training_reserve = 0.7

	#A list of alphas to experiment on when using a new model
	alpha = [1e9,1e8,1e7,1e6,1e5,1e4,1e3,1e2,1e1,1e0,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10]           #learning rate
	#alpha = [1e-8,1e-9,1e-10,1e-11,1e-12,1e-13,1e-14]  #More alphas to try iff the first set doesn't yield a good alpha
	my_data = pandas.read_csv('full_data.csv',sep=',')
	
	#Categorize the features
	cols = my_data.columns.tolist()
	list(map(cols.remove, ["Age Category", "Id", "Year"]))
	x = pandas.get_dummies(my_data[cols])
	
	#Extract the features from the data and generate a model
	data = numpy.array([x['ageFactor_[20,30)'],x['ageFactor_[30,40)'],x['ageFactor_[40,50)'],x['ageFactor_[50,60)'],x['ageFactor_[70,80)'],x['ageFactor_[80,90)'],
						x['temp'],
						x['num_1'],x['num_2'],x['num_3'],x['num_4'],x['num_5'],x['num_6'],x['num_7'],x['num_>7'],
						x['Sex_F'],
						x['Sex_M'],
						x['Sex_U'],
						x['day_no'],
						x['flu'],
						]).transpose()
	
	d = numpy.concatenate((data,numpy.ones((data.shape[0],1))),axis=1) #for the intercept weight

	
	y = numpy.array([x['ran_more_than_once']]).transpose()

	w = numpy.random.random((d.shape[1],1))
	
	
	#y_hat = numpy.round(sigmoid(w,d)); calculate_metrics(y_hat,y) #Use to make sure weights are training. This is the starting point
	
	numpy.random.shuffle(d)
	
	train_data = d[:int(math.ceil(training_reserve*d.shape[0]))]
	train_y = y[:int(math.ceil(training_reserve*d.shape[0]))]
	test_data = d[int(math.ceil(training_reserve*d.shape[0])):]
	test_y = y[int(math.ceil(training_reserve*d.shape[0])):]
	
	w = train(1,w,train_data,train_y)

	y_hat = numpy.round(sigmoid(w,test_data)); calculate_metrics(y_hat,test_y)
	
	
	#Use to locate a decent alpha value
	'''
	init_val = 1
	for a in alpha:
		pylab.subplot(5,4,init_val)
		train(a,w,train_data,train_y) # for Age Category
		init_val+=1
	'''
	

	#Use for k-fold crosss validation
	'''
	coll_of_weights = cross_validate(20,1e-8,w,d,y)
	
	for weight in coll_of_weights:	
		y_hat = numpy.round(sigmoid(weight,d)); calculate_metrics(y_hat,y)
	
	'''

	#Fine-tune alpha choice 
	'''
	#for a in alpha:
	#	coll = cross_validate(10,a,w,d,y) # Based on computational budget
	'''

	#Used for bootstrap validation
	'''
	my_train = partial(train,1,w)
	print(bootstrap(d,y,functional_error,[my_train],num_samples=200))
	'''
	
		
if __name__ == "__main__":
    main()