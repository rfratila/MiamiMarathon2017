import numpy
import csv
import pandas
import math
from linear_regression import bootstrap
from functools import partial
import time
from collections import OrderedDict
import pylab

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
	#numpy.random.shuffle(data)
	store = OrderedDict(error=[],iteration=[])
	for i in xrange(1000):
		store['error'].append(error_function(y,weight,data)[0][0])
		store['iteration'].append(i)
		weight = get_new_weight(alpha,y,weight,data)
	'''
	pylab.plot(store['iteration'],store['error'], '-ro',label='Training Error')
	pylab.xlabel("Iteration")
	pylab.ylabel("Error")
	pylab.legend(loc='upper right')
	pylab.title('Alpha is %.2g'%alpha)
	print 'Most recent error:',store['error'][-1]
	print 'Standard deviation between iterations 50 and 100:', numpy.std(store['error'][len(store['error'])/2:])
	
	return weight
	'''
	return lambda in_data: sigmoid(weight,in_data)

#K-Fold cross validation implementation
def cross_validate(k,alpha,weight,data,y):
	numpy.random.shuffle(data)
	
	chunk = data.shape[0]/k
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
		for index,test_case in enumerate(test_data):
			error += error_function(test_y[index],trained_weight,test_case)
			#print error_function(test_y[index],trained_weight,test_case)
		
		error /= test_data.shape[0]
		avg_error += error
		i +=1

	avg_error /= (data.shape[0]/k)   #Divide by the total amount of folds trained on

	print("Average error of",avg_error,"where the alpha is", alpha, "in",time.time()-start_time,"seconds")

	return collection_weights

def calculate_metrics(y_hat,y):
	TP = numpy.sum(numpy.logical_and(y_hat == 1, y ==1))
	TN = numpy.sum(numpy.logical_and(y_hat == 0, y ==0))
	FP = numpy.sum(numpy.logical_and(y_hat == 1, y ==0))
	FN = numpy.sum(numpy.logical_and(y_hat == 0, y ==1))

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

def main():
	training_reserve = 0.7
	validation_reserve = 0.2
	testing_reserve = 0.1
	alpha = [1e9,1e8,1e7,1e6,1e5,1e4,1e3,1e2,1e1,1e0,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10]           #learning rate
	my_data = pandas.read_csv('full_data.csv',sep=',')
	'''
	data = pd.read_csv("full_data.csv")
    data['Year'] = data["Year"].astype('category', ordered=True)
    cols = data.columns.tolist()
    list(map(cols.remove, ["Age Category", "Id", "Year"]))
    x = pd.get_dummies(data[cols])
    cols = x.columns.tolist()
	'''
	
	#data = numpy.array([my_data['Age Category']]).transpose()
	data = numpy.array([my_data[my_data['Year'] == 2016]['Age Category'],
						my_data[my_data['Year'] == 2016]['temp'],
						my_data[my_data['Year'] == 2016]['day_no']]).transpose()

	#data = numpy.random.random((data.shape))
	d = numpy.concatenate((data,numpy.ones((data.shape[0],1))),axis=1) #for the intercept weight

	w = numpy.random.random((d.shape[1],1))
	y = numpy.array([my_data[my_data['Year'] == 2016]['ran_more_than_once']]).transpose()
	#y = numpy.ones((y.shape))
	my_train = partial(train,1e-8,w)
	'''
	y_hat = numpy.round(sigmoid(w,d))
	calculate_metrics(y_hat,y)
	
	#train(0.0001,w,d,y)
	init_val = 1
	for a in alpha:
		pylab.subplot(5,4,init_val)
		train(a,w,d,y) # for Age Category
		init_val+=1
	pylab.show()
	import pudb; pu.db
	#for a in alpha:
	#	coll = cross_validate(10,a,w,d,y)
	'''
	print(bootstrap(d,y,functional_error,[my_train],num_samples=200))
	
	
		
if __name__ == "__main__":
    main()