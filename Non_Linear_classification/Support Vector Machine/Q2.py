import sys
import matplotlib.pyplot as plt
from copy import deepcopy
import random 
import math
import csv
import numpy as np
from sklearn import svm, datasets
from numpy import array
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
import itertools

def plotSVMClassifier(dataset):
	(data,label) = getDataSetLabelTup(dataset)
	X= data
	y= label  
	h = .02  # step size in the mesh

	# we create an instance of SVM and fit out data. We do not scale our
	# data since we want to plot the support vectors
	C = 1.0  # SVM regularization parameter
	svc = svm.SVC(kernel='linear', C=C).fit(X, y)
	rbf_svc = svm.SVC(kernel='rbf', gamma=0.10000000000000001, C=1000000.0).fit(X, y)
	poly_svc = svm.SVC(kernel='poly', degree=2, C=3.0).fit(X, y)
	lin_svc = svm.LinearSVC(C=C).fit(X, y)

	# create a mesh to plot in
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
		             np.arange(y_min, y_max, h))

	# title for the plots
	titles = ['SVC with linear kernel',
		  'LinearSVC (linear kernel)',
		  'SVC with RBF kernel',
		  'SVC with polynomial (degree 2) kernel']
	#print (poly_svc)
	for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
	    # Plot the decision boundary. For that, we will assign a color to each
	    # point in the mesh [x_min, x_max]x[y_min, y_max].
	    plt.subplot(2, 2, i + 1)
	    plt.subplots_adjust(wspace=0.4, hspace=0.4)

	    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

	    # Put the result into a color plot
	    Z = Z.reshape(xx.shape)
	    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

	    # Plot also the training points
	    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
	    plt.xlabel('Sepal length')
	    plt.ylabel('Sepal width')
	    plt.xlim(xx.min(), xx.max())
	    plt.ylim(yy.min(), yy.max())
	    plt.xticks(())
	    plt.yticks(())
	    plt.title(titles[i])
	plt.show()

def plotSVMPloyKernalClassifier(dataset,CVal,degInput):
	(data,label) = getDataSetLabelTup(dataset)
	print data
	print label
	X= data
	y= label  
	h = .02  # step size in the mesh
	# we create an instance of SVM and fit out data. We do not scale our
	# data since we want to plot the support vectors
	poly_svc = svm.SVC(kernel='poly', degree=degInput, C=CVal).fit(X, y)
	
	# create a mesh to plot in
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
		             np.arange(y_min, y_max, h))
	# title for the plots
	titles = ['SVC with polynomial kernel']
	#plt.subplot(2, 2, 1 + 1)
	#plt.subplots_adjust(wspace=0.4, hspace=0.4)

	Z = poly_svc.predict(np.c_[xx.ravel(), yy.ravel()])

	# Put the result into a color plot
	Z = Z.reshape(xx.shape)
	plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

	# Plot also the training points
	plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
	plt.xlabel('Sepal length')
	plt.ylabel('Sepal width')
	plt.xlim(xx.min(), xx.max())
	plt.ylim(yy.min(), yy.max())
	plt.xticks(())
	plt.yticks(())
	plt.title(titles[0])
	plt.show()

def plotSVMGaussRBFKernalClassifier(dataset,CVal,gammaIn):
	(data,label) = getDataSetLabelTup(dataset)
	X= data
	y= label  
	h = .02  # step size in the mesh
	# we create an instance of SVM and fit out data. We do not scale our
	# data since we want to plot the support vectors
	rbf_svc = svm.SVC(kernel='rbf', gamma=gammaIn, C=CVal).fit(X, y)
	
	# create a mesh to plot in
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
		             np.arange(y_min, y_max, h))
	# title for the plots
	titles = ['SVC with Gaussian(RBF) kernel']
	Z = rbf_svc.predict(np.c_[xx.ravel(), yy.ravel()])

	# Put the result into a color plot
	Z = Z.reshape(xx.shape)
	plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

	# Plot also the training points
	plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
	plt.xlabel('Sepal length')
	plt.ylabel('Sepal width')
	plt.xlim(xx.min(), xx.max())
	plt.ylim(yy.min(), yy.max())
	plt.xticks(())
	plt.yticks(())
	plt.title(titles[0])
	plt.show()

def getDataSetForPlot(dataset): #label -1,1
	positivex = []
	positivey = []
	negativex = []
	negativey = []
	for row in dataset:
		if row[2] == 1:
			positivex.append(row[0])
			positivey.append(row[1])
		else:
			negativex.append(row[0])
			negativey.append(row[1])
		
	return (positivex,positivey,negativex,negativey) 

def getDataSetLabel(dataset): #label -1,1
	data = []
	label = []
	for row in dataset:
		data.append([row[0],row[1]])
		label.append(row[2])	
	#data = np.array(data)
	#data = data.reshape((len(data), 1))
	return (data,label) 

def getDataSetLabelTup(dataset): #label -1,1
	data = []
	label = []
	for row in dataset:
		data.append([row[0],row[1]])
		label.append(row[2])	
	data = np.array(data)
	return (data,label) 


#create dataset from CSV file 
def loadCsv(filename):
	lines = csv.reader(open(filename, "rb"))
	dataset = list(lines)
	datasetCopy=[]
	dataset = deepcopy(dataset[1:])
	#print dataset
	index = 0
	for i in range(1, len(dataset)):
    		temp = []
		datarow = dataset[i]
		index = 0
		temp.insert(index,float(datarow[0]));index = index +1
		temp.insert(index,float(datarow[1]));index = index +1
		temp.insert(index,float(datarow[2]));
		datasetCopy.append(temp)
	return datasetCopy

def datasetup(filename):
	inputList = loadCsv(filename)
	return inputList

def computeAccuracySDUtil(c,d,dataset,polyOrRbf,epochCount=30,num_folds=10):
	accList = []
	sdList =  []
	accSdList= []
	for i in range(epochCount):
		(accuracyMean,accuracySD) = computeAccuracySD(c,d,dataset,polyOrRbf,num_folds=10)
		accList.append(accuracyMean)
		sdList.append(accuracySD)
		accSdList.append([accuracyMean,accuracySD])
	avg= np.average(accList)*100
	sd = np.std(sdList)
	avg = float("{0:.2f}".format(float(avg)))
	sd = float("{0:.3f}".format(float(sd)))
	return (avg,sd)

def computeAccuracySD(c,d,dataset,polyOrRbf,num_folds=10): #num_folds = k, polyOrRbf = True for poly
	total = len(dataset)
	featureCount = len(dataset[0]) - 1
	subset_size = len(dataset)/num_folds
	accuracy = 0.0
	random.shuffle(dataset)	
	accuracyList = []
	for i in range(num_folds):
		testSet = dataset[i*subset_size:][:subset_size]
		trainingSet = dataset[:i*subset_size] + dataset[(i+1)*subset_size:]
		if polyOrRbf:
			(success,wrong) = getAccuracyPolyKernal(trainingSet,testSet,c,d)
		else :
			(success,wrong) = getAccuracyRbfKernal(trainingSet,testSet,c,d)		
		t = success + wrong 
		accuracy = float(success) / t
		accuracyList.append(accuracy)
	accuracyMean = np.average(accuracyList)
	accuracySD = np.std(accuracyList)
	#print "accuracyMean is ",accuracyMean 
	
	return (accuracyMean,accuracySD)

def getAccuracyPolyKernal(trainingSet,testSet,CVal=1.0,dim=3): #d = dimension 
	(X,y) = getDataSetLabel(trainingSet)
	model = svm.SVC(kernel='poly', C = CVal, degree = dim)
	model.fit(X,y)
	sucess = 0
	for row in testSet:
		predictVal = model.predict([row[0],row[1]])
		if predictVal == row[2]:
			sucess = sucess + 1
	return (sucess,len(testSet)-sucess)

def bestRbfParameter(dataset):
	(X,y) = getDataSetLabel(dataset)
	C_range = np.logspace(-2, 10, 13)
	gamma_range = np.logspace(-9, 3, 13)
	param_grid = dict(gamma=gamma_range, C=C_range)
	cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
	grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
	grid.fit(X, y)
	print("The best parameters are %s with a score of %0.2f"
	      % (grid.best_params_, grid.best_score_))
	(CVal, gamma) = (grid.best_params_.get('C'), grid.best_params_.get('gamma'))
	return (CVal, gamma) 
def getAccuracyRbfKernal(trainingSet,testSet,CVal=1.0,gammaVal=0.101): #d = dimension 
	(X,y) = getDataSetLabel(trainingSet)
	C_range = np.logspace(-2, 10, 13)
	gamma_range = np.logspace(-9, 3, 13)
	clf = svm.SVC(kernel='rbf', C = CVal,gamma=gammaVal)
	clf.fit(X,y)
	sucess = 0
	for row in testSet:
		predictVal = clf.predict([row[0],row[1]])
		if predictVal == row[2]:
			sucess = sucess + 1
	return (sucess,len(testSet)-sucess)

def PolynomialKernal():
	filename= 'Data_SVM.csv'
	num_folds=10
	epochCount = 30
	c = [1.0,2.0,3.0]
	d = [2,3,4]
	CD = list(itertools.product(c, d))
	dataset = datasetup(filename)
	accuracySDMatrix = [] 
	print "Enter your choice: 1. Generate chart for Accuracy and SD with C and Degree \n 2.Generate Classifier with best C and Degree 			combination "
	num = int(raw_input())
	if num ==1: 
		for row in CD:
			(accuracyMean,accuracySD) = computeAccuracySDUtil(row[0],row[1],dataset,True,epochCount,num_folds) 
			accuracySDMatrix.append((accuracyMean,accuracySD))
		print "CD and Accuracy matrix is ",CD, accuracySDMatrix
		#plotAccuracySDBarchart(accuracySDMatrix,"PolynomialKernal",CD,"C and Degree")
		
	elif num == 2:
		plotSVMPloyKernalClassifier(dataset,3.0,2)
		
	
def GaussianKernal():
	filename= 'Data_SVM.csv'
	num_folds=10
	epochCount = 30
	c = [1.0,2.0,3.0]
	d = [0.10001,0.7,5.0] #d is gamma
	CD = list(itertools.product(c, d)) 
	accuracySDMatrix = [] 
	dataset = datasetup(filename)
	print "Enter your choice: 1. Generate chart for Accuracy and SD with C and Degree \n 2.Generate Classifier with best C and gamma 			combination "
	num = int(raw_input())
	if num ==1: 
		for row in CD:
			(accuracyMean,accuracySD) = computeAccuracySDUtil(row[0],row[1],dataset,False,epochCount,num_folds) 
			accuracySDMatrix.append((accuracyMean,accuracySD))
		print "CD and Accuracy matrix is ",CD, accuracySDMatrix
		plotAccuracySDBarchart(accuracySDMatrix,"GaussianKernal(rbf)",CD,"C and gamma")
	elif num ==2:
		(CVal, gamma)= bestRbfParameter(dataset)
		#print (CVal, gamma)
		plotSVMGaussRBFKernalClassifier(dataset,CVal,gamma)

def plotAccuracySDBarchart(accuracySDMatrix,datasetInfo,CD,varyingParam):
	accuracy, sd =  zip(*accuracySDMatrix)
	ncombination = len(CD)
	sd = [i * 100 for i in sd]
	accuracy_tup= tuple(accuracy)
	sd_tup = tuple(sd)
	fig, ax = plt.subplots()
	ax.set_ylim([0, 100])
	index = np.arange(ncombination)
	bar_width = 0.35
	opacity = 0.4
	error_config = {'ecolor': '0.3'}
	rects1 = plt.bar(index, accuracy_tup, bar_width,
		         alpha=opacity,
		         color='b',
		         error_kw=error_config,
		         label='Accuracy')

	rects2 = plt.bar(index + bar_width, sd_tup, bar_width,
		         alpha=opacity,
		         color='r',
		         error_kw=error_config,
		         label='Std. Deviation(Multiplied by 100)')

	plt.xlabel('varying  parameter '+varyingParam)
	plt.ylabel('Accuracy and SD ')
	plt.title('Accuracy and SD with ' + datasetInfo)
	plt.xticks(index + bar_width / 2, tuple(CD))
	plt.legend()
	plt.tight_layout()
	plt.show()

def KernalComparison():
	filename= 'Data_SVM.csv'
	plotSVMClassifier(datasetup(filename))

def PlotDatapoint(dataset):
	(data,label) = getDataSetLabelTup(dataset)
	X= data
	y= label  
	h = .02  # step size in the mesh
	# we create an instance of SVM and fit out data. We do not scale our
	# data since we want to plot the support vectors
	#rbf_svc = svm.SVC(kernel='rbf', gamma=gammaIn, C=CVal).fit(X, y)
	
	# create a mesh to plot in
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
		             np.arange(y_min, y_max, h))
	# title for the plots
	titles = ['Data points(SVM classification)']
	#Z = rbf_svc.predict(np.c_[xx.ravel(), yy.ravel()])

	# Put the result into a color plot
	#Z = Z.reshape(xx.shape)
	#plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

	# Plot also the training points
	plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
	plt.xlabel('Sepal length')
	plt.ylabel('Sepal width')
	plt.xlim(xx.min(), xx.max())
	plt.ylim(yy.min(), yy.max())
	plt.xticks(())
	plt.yticks(())
	plt.title(titles[0])
	plt.show()
def PlotData():
	filename= 'Data_SVM.csv'
	PlotDatapoint(datasetup(filename))

# map the inputs to the function blocks
options = {
		1 : PolynomialKernal,
       		2 : GaussianKernal,
		3 : KernalComparison,
		4 : PlotData
	}
#start
if __name__ == '__main__':
	print "1. Polynomial Kernal \n2. Gaussian kernal\n 3.plotSVMClassifier with various kernal \n4 : PlotData"
	print "Enter your choice:\t"
	num = int(raw_input())
	options[num]()
