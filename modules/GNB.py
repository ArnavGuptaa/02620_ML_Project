import numpy as np
import math
import random
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report

class GNB:
    
	def __init__(self):
		self.trainParam = {}


	# Summarize Feature
	def summarizeFeature(self,featureArr):
		# Everything should be numpy array type
		# Data Mean
		featureMean = sum(featureArr)/len(featureArr)
		# Data STD - sample
		featureSTD = sum((featureArr - featureMean) ** 2)/(len(featureArr)-1)
		return [featureMean,featureSTD**.5]

	# Summarize Data
	def SummarizeData(self,features):
		# 1st column in mean and 2nd column is std
		summaries = np.empty(0)
		for feature in np.transpose(features):
		    summaries = np.append(summaries, self.summarizeFeature(feature))
		summaries = np.reshape(summaries,(-1,2))
		return summaries

	# Calculate prior probability
	def GetPrior(self,Y):
		# Get the number of classes present
		classes = np.unique(Y)
		#     n_classes = len(classes)
		# Get total numbers for each class
		n_in_c = np.empty(0)
		for c in classes:
		    n_in_c = np.append(n_in_c, len(np.where(Y==c)[0]))
		# Total number of targets
		t = len(Y)
		# Prior probabilities' calculation
		priors = n_in_c / t
		return np.array(priors)#[0.5, 0.5]

	# Calculate Gaussian probability density function 
	def GaussPDF(self,val, mean, std):
		return ((1/(((2*math.pi)**.5)*std)) * ((math.e)**((-1/2)*(((val-mean)/std)**2))))

	# Calculate likelihood of the test data with respect to each class
	def EstimateLikelihood(self,test, param):
		n_classes = len(param)
		likelihoods = {}
		for c in list(param.keys()):
		    lik = 1
		    for f in range(len(test)):
		        c_mean = param[c]['sumStats'][f][0]
		        c_std = param[c]['sumStats'][f][1]
		        lik = lik * self.GaussPDF(test[f], c_mean, c_std)
		#             print(c, c_mean, c_std, test[f], GaussPDF(test[f], c_mean, c_std))
		    likelihoods[c] = lik
		#         print(lik)
		return likelihoods

	# Calculate the Posterior Probabilities
	def EstimatePosterior(self,likelihoods, param):
	    posteriors = {}
	    # Calculate Marginal Probability
	    margProb = 0
	    for c in list(param.keys()):
	        margProb += param[c]['prior'] * likelihoods[c]
	    # Calculate Posterior Probability
	    for c in list(param.keys()):
	        posteriors[c] = param[c]['prior'] * likelihoods[c] / margProb
	    return posteriors

	# Training the model
	def TrainModel(self,X, Y):    
	    # Get prior probability for each class
	    priors = self.GetPrior(Y)
	#     print("Priors = ",priors)
	    # Get the number of classes present
	    classes = np.unique(Y)
	    trainParam = {}
	    # Get summary data for each class 
	    for c in classes:
	        cIdx = np.where(Y==c)[0]
	        cX = X[cIdx,:]
	#         print(c, cX.shape)
	        cSummary = self.SummarizeData(cX) 
	        trainParam[c] = {
	            'prior':priors[np.where(classes==c)[0]],
	            'sumStats': cSummary
	        }

	    return trainParam

	# Testing a single data point
	def TestData(self,test, param):
	    # Get the likelihoods for every class
	    likelihoods = self.EstimateLikelihood(test, param)
	    # Get the posterior probabilities for every class
	    posteriors = self.EstimatePosterior(likelihoods, param)
	    # Get the class for the Maximum Posterior Probability
	    maxClass = max(posteriors, key = posteriors.get)
	    return maxClass

	# Testing the model with multiple data points
	def TestModel(self, X):
	    param = self.trainParam
	    predictedClass = np.empty(0)
	    for test in X:
	        predictedClass = np.append(predictedClass, self.TestData(test, param))

	    return predictedClass

	# Get the Accuracy of the Model
	def ModelAccuracy(self,predictedVal, trueVal):
	    correct = 0
	    for p, t in zip(predictedVal, trueVal):
	        if p == t:
	            correct += 1
	    acc = correct / len(trueVal)
	    return acc


	def SplitDataset(self,X,Y,sratio):
	    # Split Dataset
	    splitPercent = sratio
	    self.size_trainData = int(len(X)*splitPercent)
	    size_testData = len(X)-size_trainData

	    # Randomly select indices
	    trainIdx = random.sample(range(len(X)), size_trainData)

	    # Get random training data
	    trainX = X
	    trainX = trainX[trainIdx,:]

	    # Get the rest of the testing data
	    testIdx = list(set(range(len(X))) - set(trainIdx))
	    testX = X
	    testX = testX[testIdx,:]

	    # Split Y
	    trainY = Y[trainIdx]
	    testY = Y[testIdx]

	    return [trainX, trainY, testX, testY]


	def fit(self,X,Y):
	    # Train the Model
	    # trainX, trainY, self.testX, self.testY = self.SplitDataset(X,Y,0.8)
	    self.trainParam = self.TrainModel(X, Y)

	def predict(self, X):
	    return self.TestModel(X) 

	def cross_valid(self,X,y):
		new_label = {'Glioma':1,'Glioblastoma':1, 'A+O':2, 'Astrocytoma':3, 'Oligodendroglioma':4, 'Normal':5}
		Ypre = y.apply(lambda x: new_label[x])
		y = Ypre.to_numpy()

		master_dict = {'accuracy':[],'precision':[],'recall':[],'f1':[]}
		indices = random.sample(range(len(X)), len(X))
		X = X[indices]
		y = y[indices]
		kf = KFold()
		for train_index, test_index in kf.split(X):
		    X_train, X_test = X[train_index], X[test_index]
		    y_train, y_test = y[train_index], y[test_index]
		    model = self
		    model.fit(X_train, y_train)
		    y_pred = model.predict(X_test)
		    report = classification_report(y_test,y_pred,output_dict=True)
		    master_dict['accuracy'].append(report['accuracy'])
		    master_dict['precision'].append(report['macro avg']['precision'])
		    master_dict['recall'].append(report['macro avg']['recall'])
		    master_dict['f1'].append(report['macro avg']['f1-score'])
		result ={}
		result['accuracy'] = np.mean(master_dict['accuracy'])
		result['precision'] = np.mean(master_dict['precision'])
		result['recall'] = np.mean(master_dict['recall'])
		result['f1'] = np.mean(master_dict['f1'])
		return result





