import sys
import matplotlib.pyplot as plt
import numpy as np
#from matplotlib.lines import Line2D

# Make a prediction with weights
def compute(row, weights):
	bias = weights[2]
	output = bias
	#output = (w1 * X1) + (w2 * X2) + bias
	for i in range(len(row)-1):
		output += weights[i] * row[i]
		##print "output is",output
	return 1 if output > 0 else 0

#Update weight and bias 
def updateWeight(weights,x):
	#update bias
	weights[2] = weights[2] + x[2]
	#update weight part w1, w2
	for i in range(len(x)-1):
		weights[i] = weights[i] + x[i]
	return weights

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
def plotCoordinates(dataset,weightPlot):
	XList1 =[]
	YList1 =[]
	XList2 =[]
	YList2 =[]	
	count = 0
	boundX = -8
	boundY = 10
	#compute classifier co-ordinates
	##print "2: Weight plot is ",weightPlot
	x1 = - (weightPlot[2]/weightPlot[1])
	y1 = 0
	x2 = 0
	y2 = - (weightPlot[2]/weightPlot[0])
	#print x1 , y2
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
	#plt.plot([-3.92,0,1.19,3.17,11.31],[-6,-0.94,0,2,8])
	plt.plot(plotTup[0],plotTup[1])
	#plt.plot([x2,y2],[x1,y1])
	
	#plt.plot([-8,1.64,0,8],[-7.85,0,-1.34,5.17])
	plt.show()
	
def findPerceptronClassifier(dataset,weights):
	flag = True
	epoch = 0
	retList = []
	#lastWeight = []
	while(flag):
		flag = False
		epoch = epoch + 1  
		#print("\nepoch = epoch + 1 is %d\n",epoch)
		for row in dataset:
			predicted_val = compute(row, weights)
			error1 = row[-1] - predicted_val
			#print("Expected=%d, Predicted=%d" % (row[-1], predicted_val))
			if predicted_val <= 0:
				#update weights
				weights = updateWeight(weights,row)
				lastWeight = weights
				flag = True
				##print "Last row is",row[0],row[1],row[2]
				##print("Nik:- weight is%d - %d - %d ",weights[0],weights[1],weights[2])
	retList.append(epoch)
	#print "Weight is ",weights
	#print "last Weight is ",lastWeight
	
	retList.append(weights)	
	return retList

#Update weight and bias 
def updateWeight1(weights,x,l_rate,error):
	#update bias
	weights[2] = weights[2] + x[2] + l_rate * error
	#update weight part w1, w2
	for i in range(len(x)-1):
		weights[i] = weights[i] + l_rate * error * x[i]
	return weights

	
def findPerceptronClassifier1(dataset,weights):
	flag = True
	epoch = 0
	retList = []
	l_rate = 0.2
	count = 0
	#lastWeight = []
	while(flag):
		#flag = False
		epoch = epoch + 1  
		#print("\nepoch = epoch + 1 is %d\n",epoch)
		count = 0
		for row in dataset:
			
			predicted_val = compute(row, weights)
			error = row[-1] - predicted_val
			#print("Expected=%d, Predicted=%d" % (row[-1], predicted_val))			
			#update weights
			if error != 0:
				weights = updateWeight1(weights,row,l_rate,error)
				count = count + 1
			lastWeight = weights
			#print "Last row is",row[0],row[1],row[2]
			#print("Nik:- weight is%d - %d - %d ",weights[0],weights[1],weights[2])
		if error == 0 and count == 0:			
				flag = False
		else:
			flag = True
	retList.append(epoch)
	#print "Weight is ",weights
	#print "last Weight is ",lastWeight
	
	retList.append(weights)	
	return retList
	
# Input dataset for classifier 
datasetC1C2 =[[0.1,1.1,0], [6.8 ,7.1,0], [-3.5 ,-4.1,0], [2.0 ,2.7,0] , [4.1 ,2.8,0] ,
	  [3.1 ,5.0,0], [-0.8 ,-1.3,0],[0.9 ,1.2,0], [5.0 ,6.4,0], [3.9, 4.0,0],
	  [7.1 ,4.2,1], [-1.4, -4.3,1],[4.5 ,0.0,1 ], [6.3 ,1.6,1 ],[4.2 ,1.9,1 ], 
	  [1.4 ,-3.2,1], [2.4 ,-4.0,1 ],[2.5 ,-6.1,1 ],[8.4 ,3.7,1], [4.1 ,-2.2,1]]


datasetC2C3 = [[-3.0 , -2.9,0], [0.5,  8.7,0], [2.9 , 2.1,0], [-0.1,  5.2,0], [-4.0 , 2.2,0], [-1.3,  3.7,0], 
	[-3.4,  6.2,0], [-4.1,  3.4,0], [-5.1,  1.6,0], [1.9 , 5.1,0],[7.1 ,4.2,1], 
	[-1.4, -4.3,1],[4.5 ,0.0,1 ], [6.3 ,1.6,1 ],[4.2 ,1.9,1 ], 	  
	[1.4 ,-3.2,1], [2.4 ,-4.0,1 ],[2.5 ,-6.1,1 ],[8.4 ,3.7,1], [4.1 ,-2.2,1]]

#initialize inital weight and bias
initial_weights = [0,0,0]
#Iteration count 
epoch = 0
outList = []
def C1C2Classifier():
	#Iteration count for convergence - Dataset C1 and C2 	
	outList = findPerceptronClassifier1(datasetC1C2,initial_weights)
	epoch = outList[0]
	weightPlot = outList[1]
	#print "Number of iterations required for convergence (class C1 and C2): ",epoch
	##print "Weight plot is ",weightPlot
	plotCoordinates(datasetC1C2,weightPlot)
	
def C2C3Classifier():
    	#Iteration count for convergence - Dataset C2 and C3 	
	outList = findPerceptronClassifier1(datasetC2C3,initial_weights)
	epoch = outList[0]
	weightPlot = outList[1]
	#print "Number of iterations required for convergence (class C1 and C2): ",epoch
	##print "Weight plot is ",weightPlot
	plotCoordinates(datasetC2C3,weightPlot)

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

