import numpy as np
from sklearn.cross_validation import train_test_split 
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer
import csv

def sigmoid (x): 
	return 1/(1 + np.exp(-x))      # activation function
def sigmoid_derivative(x): 
	return x * (1 - x)             # derivative of sigmoid # weights on layer inputs
                                                
class NeuralNetwork:
	def __init__(self, layers, activation='sigmoid'):
		if activation == 'sigmoid':
			self.activation = sigmoid
			self.activation_deriv = sigmoid_derivative

        	self.weights = []
		for i in range(1, len(layers) - 1):
			self.weights.append((2*np.random.random((layers[i - 1] + 1, layers[i]
						))-1)*0.25)
		self.weights.append((2*np.random.random((layers[i] + 1, layers[i +
					1]))-1)*0.25)	

	def model(self, X, y, learning_rate=0.2, epochs=10000):
		X = np.atleast_2d(X)
		temp = np.ones([X.shape[0], X.shape[1]+1])
		temp[:, 0:-1] = X  # adding the bias unit to the input layer
		X = temp
		y = np.array(y)
		#print X,y
		for k in range(epochs):
			i = np.random.randint(X.shape[0])
			a = [X[i]]

			for l in range(len(self.weights)):
				hidden_inputs = np.ones([self.weights[l].shape[1] + 1])
				hidden_inputs[0:-1] = self.activation(np.dot(a[l], self.weights[l]))
				a.append(hidden_inputs)
			error = y[i] - a[-1][:-1]
			deltas = [error * self.activation_deriv(a[-1][:-1])]
			l = len(a) - 2

			# The last layer before the output is handled separately because of
			# the lack of bias node in output
			deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_deriv(a[l]))

			for l in range(len(a) -3, 0, -1): # we need to begin at the second to last layer
				deltas.append(deltas[-1][:-1].dot(self.weights[l].T)*self.activation_deriv(a[l]))

			deltas.reverse()
			for i in range(len(self.weights)-1):
				layer = np.atleast_2d(a[i])
				delta = np.atleast_2d(deltas[i])
				self.weights[i] += learning_rate * layer.T.dot(delta[:,:-1])
			# Handle last layer separately because it doesn't have a bias unit
			i+=1
			layer = np.atleast_2d(a[i])
			delta = np.atleast_2d(deltas[i])
			self.weights[i] += learning_rate * layer.T.dot(delta)
		#print self.weights

	def predict(self, x):
		a = np.array(x)
		for l in range(0, len(self.weights)):
			temp = np.ones(a.shape[0]+1)
			temp[0:-1] = a
			a = self.activation(np.dot(temp, self.weights[l]))
		return a

def doProcess(data,label,flag):
	Nhidden = [1,5,10,30,50,70,80,100,150]
	nIn = 64
	nOut = 3
	nH = 100	
	X = data
	y = label
	X -= X.min() # normalize the values to bring them into the range 0-1
	X /= X.max()
	
	for nH in Nhidden:
		nn = NeuralNetwork([nIn,nH,nOut],'sigmoid')
		X_train, X_test, y_train, y_test = train_test_split(X, y)
		labels_train = LabelBinarizer().fit_transform(y_train)
		labels_test = LabelBinarizer().fit_transform(y_test)
		weights = nn.model(X_train,labels_train,epochs=30000)
		if flag:
			print 'Weight for nH ',nH
			print  weights
		else:
			predictions = []
			for i in range(X_test.shape[0]):
			    o = nn.predict(X_test[i] )
			    predictions.append(np.argmax(o))
			print "nH is ", nH
			print classification_report(y_test,predictions)	
	
#create dataset from CSV file 
def loadCsv(filename):
	threeDigitList=['0','1','2']
	lines = csv.reader(open(filename, "rb"))
	dataset = list(lines)
	#print len(dataset)
	datasetCopy=[]
	index = 0
	for i in range(len(dataset)):
    		temp = []
		datarow = dataset[i]		
		index = 0
		if datarow[len(datarow)-1] in threeDigitList:
			datasetCopy.append([int(i) for i in datarow])
	return datasetCopy

def getDataSetLabel(dataset): #label -1,1
	data = []
	label = []
	for row in dataset:
		label.append(row[len(row)-1])
		del row[-1:]
		data.append([i for i in row])#lst[:n-1] remove last n element 		
	#data = np.array(data)
	#print data[0]
	return (data,label) 

def datasetup(filename):
	inputList = loadCsv(filename)
	return inputList

def doDataload():
	filename = 'optdigits-tra.csv'
	dataset = datasetup(filename)
	(data,labels)  = getDataSetLabel(dataset)
	return (data,labels)
#start
if __name__ == '__main__':
	(X,y)=doDataload()
	X = np.array(X)
	y=  np.array(y)
	doProcess(X,y,False) #weight print True, Accuracy False


