import sys
import matplotlib.pyplot as plt
import pylab as pl
import numpy as np

# Make a prediction with weights
def compute(row, weights):
	bias = weights[2]
	output = bias
	for i in range(len(row)-1):
		output += weights[i] * row[i]
	return 1.0 if output > 0 else -1.0

def getMultiplePoints(x,y,weight,boundX1,boundX2):
	x1 =[x,0]
	x2 =[0,y]
	pointsX = []
	pointsY = []
	pointsX.insert(1,y)
	pointsX.insert(2,0)
	pointsY.insert(1,0)
	pointsY.insert(2,x)
	#for boundX1
	pointsX.insert(0,boundX1)
	temp = -(weight[0]*boundX1 + weight[2])/weight[1]
	pointsY.insert(0,temp)
	#for boundX2
	pointsX.insert(3,boundX2)
	temp = -((weight[0]*boundX2) + weight[2])/weight[1]
	pointsY.insert(3,temp)	
	return (pointsX,pointsY)

#plot points 
def plotCoordinates(dataset,weightPlot,datasetInfo):
	XList1 =[]
	YList1 =[]
	XList2 =[]
	YList2 =[]	
	count = 0
	boundX = -8
	boundY = 10
	#compute classifier co-ordinates
	x1 = - (weightPlot[2]/weightPlot[1])
	y1 = 0
	x2 = 0
	y2 = - (weightPlot[2]/weightPlot[0])
	
	# compute some random point with slope as W and bias b 
	plotTup = getMultiplePoints(x1,y2,weightPlot,boundX,boundY)
	for row in dataset:
		if(count<=9):
			XList1.append(row[0])
			YList1.append(row[1])
		else:
			XList2.append(row[0])
			YList2.append(row[1])
		count = count+1
	#Draw points with red and Blue color 
	plt.plot(XList1, YList1, 'ro',XList2, YList2, 'bo')
	plt.axis([boundX, boundY, boundX, boundY])
	plt.plot(plotTup[0],plotTup[1])
	plt.title(datasetInfo)
	plt.show()
	
#Update weight and bias 
def updateWeight(weights,x,error):
	#update bias
	weights[2] = weights[2] + x[2] + error
	#update weight part w1, w2
	for i in range(len(x)-1):
		weights[i] = weights[i] + error * x[i]
	return weights

	
def findPerceptronClassifier(dataset,weights):
	flag = True
	epoch = 0
	retList = []
	count = 0
	#lastWeight = []
	while(flag):
		#flag = False
		epoch = epoch + 1  
		count = 0
		for row in dataset:
			
			predicted_val = compute(row, weights)
			error = row[-1] - predicted_val		
			#update weights
			if error != 0:
				weights = updateWeight(weights,row,error)
				count = count + 1
			lastWeight = weights
		if error == 0 and count == 0:			
				flag = False
		else:
			flag = True
	retList.append(epoch)
	retList.append(weights)	
	return retList
	
# Input dataset for classifier 
datasetC1C2 =[[0.1,1.1,-1.0], [6.8 ,7.1,-1.0], [-3.5 ,-4.1,-1.0], [2.0 ,2.7,-1.0] , [4.1 ,2.8,-1.0] ,
	  [3.1 ,5.0,-1.0], [-0.8 ,-1.3,-1.0],[0.9 ,1.2,-1.0], [5.0 ,6.4,-1.0], [3.9, 4.0,-1.0],
	  [7.1 ,4.2,1.0], [-1.4, -4.3,1.0],[4.5 ,0.0,1.0], [6.3 ,1.6,1.0],[4.2 ,1.9,1.0 ], 
	  [1.4 ,-3.2,1.0], [2.4 ,-4.0,1.0 ],[2.5 ,-6.1,1.0 ],[8.4 ,3.7,1.0], [4.1 ,-2.2,1.0]]


datasetC2C3 = [[-3.0 , -2.9,-1.0], [0.5,  8.7,-1.0], [2.9 , 2.1,-1.0], [-0.1,  5.2,-1.0], [-4.0 , 2.2,-1.0], 
		[-1.3,3.7,-1.0], [-3.4,  6.2,-1.0], [-4.1,  3.4,-1.0], [-5.1,  1.6,-1.0], [1.9 , 5.1,-1.0],
		[7.1 ,4.2,1.0], [-1.4, -4.3,1.0],[4.5 ,0.0,1.0], [6.3 ,1.6,1.0],[4.2 ,1.9,1.0 ], 
	  	[1.4 ,-3.2,1.0], [2.4 ,-4.0,1.0 ],[2.5 ,-6.1,1.0 ],[8.4 ,3.7,1.0], [4.1 ,-2.2,1.0]]

#initialize inital weight and bias
initial_weights = [0,0,0]
#Iteration count 
epoch = 0
outList = []
def C1C2Classifier():
	#Iteration count for convergence - Dataset C1 and C2 	
	outList = findPerceptronClassifier(datasetC1C2,initial_weights)
	epoch = outList[0]
	weightPlot = outList[1]
	print "Number of iterations required for convergence (class C1 and C2): ",epoch
	plotCoordinates(datasetC1C2,weightPlot,'Linear classifier for class 1 and class 2 data points')
	
def C2C3Classifier():
    	#Iteration count for convergence - Dataset C2 and C3 	
	outList = findPerceptronClassifier(datasetC2C3,initial_weights)
	epoch = outList[0]
	weightPlot = outList[1]
	print "Number of iterations required for convergence (class C1 and C2): ",epoch
	plotCoordinates(datasetC2C3,weightPlot,'Linear classifier for class 2 and class 3 data points')

# map the inputs to the function blocks
options = {
		1 : C1C2Classifier,
       		2 : C2C3Classifier,
	}

#start
if __name__ == '__main__':
	print "1. Run C1C2 classifier \n2. Run C2C3 classifier\n"
	print "Enter your choice:\t"
	num = int(raw_input())
	options[num]()

