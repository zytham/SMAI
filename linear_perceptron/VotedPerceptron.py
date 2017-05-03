import sys
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import random 
import math

# Make a prediction with weights - Voted perceptron
def predictVotedPerceptron(row, weights,bias):
	row_size = len(row)
	#Outcome of this row - Yi
	y = row[row_size-1]	
	output = bias
	for i in range(len(weights)):
		output += weights[i] * row[i]
	output = output*y	
	return 1.0 if output >= 0.0 else -1.0

#Update weight and bias - for simple perceptron
def updateWeight_bias(weights,x,error):
	lenW = len(weights)
	lenR = len(x)
	#update bias
	weights[lenW-1] = weights[lenW-1] + x[lenR-1] +  error
	#update weight part w1, w2
	for i in range(len(x)-1):
		weights[i] = weights[i] + error * x[i]
	return weights

#Make a prediction with weights - Perceptron simple
def predictSP(row, weights):
	lenW = len(weights)
	outcome = weights[lenW-1] #biasat last index
	for i in range(lenW-1): # minus 1 because of last index is bias
		outcome += weights[i] * row[i]
	return 1.0 if outcome > 0.0 else -1.0


def updateDataSetWithLabel(dataset,labelClass,file1Or2): #change label 1 = -1,label 2 = 1
	if len(labelClass) == 2: #Assume two class labels coming only 
		label1 = labelClass[0]	
		label2 = labelClass[1]
	datasetDup = []
	indexDup = 0
	if file1Or2 == 1:	
		rowIndex = 1
	elif file1Or2 == 2:
		rowIndex = 0
	rowSize = len(dataset[0])	
	for row in dataset:
		if row[rowSize -1] == label1:
			row[rowSize -1] = -1.0
		elif row[rowSize -1] == label2:
			row[rowSize -1] = 1.0
		datasetDup.insert(indexDup,deepcopy(row[rowIndex:])) # keep on adding element at start of index
	return datasetDup 

def trainPerceptronWeights(dataset,featuresCount,epochCount):
	weights = [0.0 for i in range(featuresCount+1)] # +1 for adding bias at last index
	count = 0
	for epoch in range(epochCount):
		count = 0
		for row in dataset:
			predicted_val = predictSP(row, weights)
			error = row[-1] - predicted_val
			if error != 0:
				weights = updateWeight_bias(weights,row,error)
	return weights

#Update weight and bias 
def updateWeightBias(weight,row, bias):
	row_size = len(row)
	listOut = []
	#Outcome of this row - Yi
	y = row[row_size-1]
	#update bias
	bias = bias + y 
	#update weight part
	for i in range(len(weight)):
		weight[i] = weight[i] + row[i]*y
	listOut.append(weight)
	listOut.append(bias)
	return listOut

def createWeightVector(weight,bias,c):
	weightVector = []
	weightVector.insert(0,weight)
	weightVector.insert(1,bias)
	weightVector.insert(2,c)
	return weightVector	
	
def votedPerceptronTraining(dataset,epochCount,featureLen): 
	n = 1;b = 0;c = 1;bias = 0
	m = len(dataset)
	outcome = []
	indexOut = 0
	#make weight vector 
	weight = [0 for x in range(featureLen)]
	weightVector = createWeightVector(weight,bias,c)
	outcome.insert(indexOut,deepcopy(weightVector))
	indexOut = indexOut +1
	for iter in range(epochCount):
		for row in dataset:
			predict_val = predictVotedPerceptron(row,weight,bias)
			error = row[-1] - predict_val
			if error != 0.0:
				listOut = updateWeightBias(weight,row,bias)
				#update Outcome list
				weightVector = createWeightVector(listOut[0],listOut[1],c)
				weight = listOut[0]
				bias = listOut[1]
				outcome.insert(indexOut,deepcopy(weightVector))
				indexOut = indexOut + 1
				n = n+1
				c = 1
			else:
				c = c +1
	return outcome

def predictOutcome(outcomeWeightVector,row):
	row_size = len(row)
	signList = []
	#Outcome of this row - Y is actual output
	y = row[row_size-1]
	for weightRowBias in outcomeWeightVector:
		bias = weightRowBias[1] #bias at [2]
		c = weightRowBias[2]
		weights = weightRowBias[0]
		output = bias
		for i in range(len(weights)):
			output += weights[i] * row[i]
		if output <= 0:
			output = -1
		else:
			output = 1
		output = output*c
		signList.append(output)
	Yprime = sum(signList)
	if Yprime < 0:
		Yprime = -1
	else:
		Yprime = 1
	#compare sign of Yprime and Y
	if Yprime == y:
		return True
	else:
		return False

def computeAccuracyVotedPerceptron(outcomeWeightVector,testData):
	countTotal = len(testData)
	correct = 0
	wrong = 0
	for row in testData:
		value = predictOutcome(outcomeWeightVector,row)	
		if value :
			correct = correct  +1
		else :
			wrong = wrong +1
	return (correct,wrong)

def computeAccuracySimplePerceptron(weights,testData):
	countTotal = len(testData)
	correct = 0
	wrong = 0
	rowSize = len(testData[0])
	for row in testData:
		value = predictSP(row, weights)	
		if value == row[rowSize-1] :
			correct = correct  +1
		else :
			wrong = wrong +1
	return (correct,wrong)


#read file and convert input into list
def read_file_content(filename,file1Or2):
	file_handle = open(filename)
	allLines = file_handle.readlines()
	index = 0
	inputList = []
	for eachline in allLines:
		eachline = eachline.rstrip('\n')
		data = eachline.strip().split(',')
		intList = []
		dataLen = len(data)
		if file1Or2 == 1:		
			for each in data:
				intList.append(int(each))
		elif file1Or2 == 2:
			index = 0
			for each in data:
				if index < dataLen-1:
					#intList.append(float(each))
					intList.append(float("{0:.3f}".format(float(each))))
					index = index + 1
				else:  
					intList.append(each)
        	inputList.insert(index,intList)
	return inputList

def datasetup(fileName,file1Or2):
	inputList = read_file_content(fileName,file1Or2)
	return inputList

def computeMeanAccuracyVotedPerceptron(dataset,epochCount,num_folds=10):
	total = len(dataset)
	featureCount = len(dataset[0]) -1  #featuresCount = 9 for  cancer data
	accuracy = 0.0
	subset_size = len(dataset)/num_folds
    	random.shuffle(dataset)
	for i in range(num_folds):
		testDatalist = dataset[i*subset_size:][:subset_size]
		trainingList = dataset[:i*subset_size] + dataset[(i+1)*subset_size:]
		outcomeWeightVector = votedPerceptronTraining(trainingList,epochCount,featureCount)
		(correctPrecdictCount,wrongPrecdictCount) = computeAccuracyVotedPerceptron(outcomeWeightVector,testDatalist)
		t = correctPrecdictCount + wrongPrecdictCount
		accuracy = accuracy + float(correctPrecdictCount) / t
	accuracyMean = (float(accuracy)*100)/num_folds
	accuracyMean = float("{0:.2f}".format(float(accuracyMean)))
	return accuracyMean

def computeMeanAccuracySimplePerceptron(dataset,epochCount,num_folds=10): #num_folds = k
	total = len(dataset)
	featureCount = len(dataset[0]) - 1
	subset_size = len(dataset)/num_folds
	accuracy = 0.0
	random.shuffle(dataset)
	for i in range(num_folds):
		testDatalist = dataset[i*subset_size:][:subset_size]
		trainingList = dataset[:i*subset_size] + dataset[(i+1)*subset_size:]
		trainiwdWeights	= trainPerceptronWeights(trainingList,featureCount,epochCount)		
		(correctPrecdictCount,wrongPrecdictCount) = computeAccuracySimplePerceptron(trainiwdWeights,testDatalist)
		t = correctPrecdictCount + wrongPrecdictCount
		accuracy = accuracy + (float(correctPrecdictCount) / t)
	#trainiwdWeights = float("{0:.3f}".format(float(trainiwdWeights)))
	accuracyMean = (float(accuracy)*100)/num_folds
	accuracyMean = float("{0:.2f}".format(float(accuracyMean)))
	return (accuracyMean,trainiwdWeights)


def plotAccuracyWithEpoch(accuracyWithEpochVP,accuracyWithEpochSP,datasetInfo):
	accListVP= []
	labelListVP =[]
	keylistVP = accuracyWithEpochVP.keys()
	keylistVP.sort()
	for key in keylistVP:
		accListVP.append(accuracyWithEpochVP[key])
		labelListVP.append(str(key))
	accListSP= []
	keylistSP = accuracyWithEpochSP.keys()
	keylistSP.sort()
	for key in keylistSP:
		accListSP.append(accuracyWithEpochSP[key])
	n_epoch = len(accuracyWithEpochVP)
	voted_acc= tuple(accListVP)
	simpleperc_acc = tuple(accListSP)
	
	fig, ax = plt.subplots()
	ax.set_ylim([0, 100])
	index = np.arange(n_epoch)
	bar_width = 0.35

	opacity = 0.4
	error_config = {'ecolor': '0.3'}

	rects1 = plt.bar(index, voted_acc, bar_width,
		         alpha=opacity,
		         color='b',
		         error_kw=error_config,
		         label='Voted perceptron')

	rects2 = plt.bar(index + bar_width, simpleperc_acc, bar_width,
		         alpha=opacity,
		         color='r',
		         error_kw=error_config,
		         label='Simple perceptron')

	plt.xlabel('Epoch Count')
	plt.ylabel('Accuracy')
	plt.title('Accuracy VS Epoch for Voted and simple perceptron with ' + datasetInfo)
	
	#plt.xticks(index + bar_width / 2, ('10','15', '20', '25', '30', '35','40', '45', '50'))
	plt.xticks(index + bar_width / 2, tuple(labelListVP))
	plt.legend()
	plt.tight_layout()
	plt.show()

def DatasetCancer():
	filename1 = "Data/breast-cancer-wisconsin.data.txt"
	labelClass = [2,4]
	k_fold = 10
	epochList = [10,15, 20, 25, 30, 35,40, 45, 50]
	accuracyWithEpochVP = {}
	accuracyWithEpochSP = {}
	#setup data from file 
	datasetBSW = datasetup(filename1,1)
	
	#create a copy of dataset for Voted Perceptron	
	datasetVP = deepcopy(datasetBSW)

	#Update label with appropriate value
	datasetSP= updateDataSetWithLabel(datasetBSW,labelClass,1)
	datasetVP= updateDataSetWithLabel(datasetVP,labelClass,1)
	
	for epochCount in epochList:
		accuracyMean = computeMeanAccuracyVotedPerceptron(datasetVP,epochCount,k_fold)
		accuracyWithEpochVP[epochCount] = accuracyMean
	
	epochWeightvectorMap = {}
	#for simple perceptron 
	for epochCount in epochList:
		#k= 10, report 10-fold cross validation accuracies
		(accuracyMean,trainiwdWeights) = computeMeanAccuracySimplePerceptron(datasetSP,epochCount,k_fold)
		accuracyWithEpochSP[epochCount] = accuracyMean
		epochWeightvectorMap[epochCount] = trainiwdWeights
	
	#weight vector learned with epoch 
	keylistSP = epochWeightvectorMap.keys()
	keylistSP.sort()
	for key in keylistSP:
		print "Epoch :%d : weight learned\n ",key,epochWeightvectorMap[key]
	
	#plot accuracy vs Epoch count
	plotAccuracyWithEpoch(accuracyWithEpochVP,accuracyWithEpochSP,'breast-cancer-wisconsin.data')

def DatasetIomosphere():
	filename2 = "Data/ionosphere.data.txt"
	labelClass = ['g','b']
	k_fold = 10
	epochList = [10,15, 20, 25, 30, 35,40, 45, 50]
	accuracyWithEpochVP = {}
	accuracyWithEpochSP = {}
	#setup data from file 
	datasetION = datasetup(filename2,2)
	#create a copy of dataset for Voted Perceptron	
	datasetVP = deepcopy(datasetION)

	#Update label with appropriate value
	datasetSP= updateDataSetWithLabel(datasetION,labelClass,2)
	datasetVP= updateDataSetWithLabel(datasetVP,labelClass,2)
	
	for epochCount in epochList:
		accuracyMean = computeMeanAccuracyVotedPerceptron(datasetVP,epochCount,k_fold)
		accuracyWithEpochVP[epochCount] = accuracyMean
	
	epochWeightvectorMap = {}
	#for simple perceptron 
	for epochCount in epochList:
		#k= 10, report 10-fold cross validation accuracies
		(accuracyMean,trainiwdWeights) = computeMeanAccuracySimplePerceptron(datasetSP,epochCount,k_fold)
		accuracyWithEpochSP[epochCount] = accuracyMean
		epochWeightvectorMap[epochCount] = trainiwdWeights
	
	#weight vector learned with epoch 
	keylistSP = epochWeightvectorMap.keys()
	keylistSP.sort()
	for key in keylistSP:
		print "Epoch :%d : weight learned\n ",key,epochWeightvectorMap[key]
	
	#plot accuracy vs Epoch count 
	plotAccuracyWithEpoch(accuracyWithEpochVP,accuracyWithEpochSP,'ionosphere.data')
	
# map the inputs to the function blocks
options = {
		1 : DatasetCancer,
       		2 : DatasetIomosphere,
	}
#start
if __name__ == '__main__':
	print "1. Epoch Vs Accuracy on dataset breast-cancer-wisconsin.data.txt \
\n2. Epoch Vs Accuracy on dataset ionosphere.data.txt \n"
	print "Enter your choice:\t"
	num = int(raw_input())
	options[num]()
	
