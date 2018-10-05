import numpy as np
from numpy.linalg.linalg import pinv
import math
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from datetime import datetime
import sys
import matplotlib.pyplot as plt
import csv
#ignore deprecation warnings
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    from sklearn.cross_validation import StratifiedShuffleSplit


#exec(open('Seeds_preprocess.py').read())
#exec(open('Iris_preprocess.py').read())
#exec(open('Transfusion_preprocess.py').read())
#exec(open('Cancer_preprocess.py').read())
exec(open('glass_preprocess.py').read())

#distance between multidimensional data points
def euclidean_distance(a, b):
	return np.linalg.norm(a-b)


# RBFNN
class RBFNetwork:
    def __init__(self, scaledData, testData, numClasses, NumOfRBF = 0):
        #set the number of RBF neurons
        self.RBFs = NumOfRBF
        self.scaledData = scaledData
        self.protos = np.zeros(shape=(0, len(self.scaledData[1])))
        self.testData = testData      
        #print (len(self.scaledData[1]))
        self.labels = self.scaledData[:,[-1]].astype(np.int)
        self.testLabels = self.testData[:,[-1]].astype(np.int)
        self.predictedLabels = np.zeros(shape=(1, 0))
        self.testLabelsLine = np.zeros(shape=(1, 0))
        self.weights = 0
        self.MSE = 0
        self.numClasses = numClasses
        self.spread = np.zeros(shape=(1, self.RBFs))
        #delete class column from dataset
        self.scaledDatanoClassColumn = np.delete(self.scaledData, len(self.scaledData[0])-1, axis=1)
        self.testDatanoClassColumn = np.delete(self.testData, len(self.testData[0])-1, axis=1)
        
        #TRANSFORM LABELS INTO PROPER FORM
        self.PredictionValue = np.zeros(shape=(0, len(np.unique(self.labels))))
        x, y = self.labels.shape
        self.ClassLabels = np.zeros(shape = (0,len(np.unique(self.labels))))
        for i in range(0,x):
            posNum = self.labels[i]
            newRow = np.zeros(shape=(1, len(np.unique(self.labels))))
            newRow[:, posNum-1] = 1
            self.ClassLabels = np.vstack([self.ClassLabels, newRow]).astype(np.int)

        x2, y2 = self.testLabels.shape
        self.Test_Labels = np.zeros(shape = (0,len(np.unique(self.labels))))
        for i in range(0,x2):
            posNum = self.testLabels[i]
            newTestRow = np.zeros(shape=(1, len(np.unique(self.labels))))
            newTestRow[:, posNum-1] = 1
            self.Test_Labels = np.vstack([self.Test_Labels, newTestRow]).astype(np.int)
        #print(self.ClassLabels.shape)
        #print(self.Test_Labels.shape)
        #exit()

    def kmeans(self, k, dataset, epsilon=0):
        num_instances, num_features = dataset.shape
        #random pick k centroids from given data
        centroids = dataset[np.random.randint(0, num_instances - 1, size=k)]
        #saving past centroids
        old_centroids = np.zeros(centroids.shape)
        #all of the data points belongig to a cluster
        currentCluster = np.zeros((num_instances, 1))
        #euclidean distance between current and old centroids, it has a value of 0 when final centroids are found
        norm = euclidean_distance(centroids, old_centroids)
        iteration = 0
        #while optimal centers are NOT found
        while norm > epsilon:
            iteration += 1
            norm = euclidean_distance(centroids, old_centroids)
            #print ("Norm: ", norm)
            old_centroids = centroids
    
            #iterate trough dataset
            for index_instance, instance in enumerate(dataset):
                dist_vec = np.zeros((k, 1))
                #iterate trough all k centroids
                for index_prototype, prototype in enumerate(centroids):
                	#compute distance between each centroid (prototype) and all data points
                    dist_vec[index_prototype] = euclidean_distance(prototype, instance)
                #add the smallest distance to a cluster
                currentCluster[index_instance, 0] = np.argmin(dist_vec)
    
            tmpCluster = np.zeros((k, num_features))
    
            #for each cluster
            for index in range(len(centroids)):
            	#all data points that belong to cluster
                instances_close = [i for i in range(len(currentCluster)) if currentCluster[i] == index]
                #calculating mean from those data points
                if dataset[instances_close].any():
                	prototype = np.mean(dataset[instances_close], axis=0)
                	tmpCluster[index, :] = prototype
    
            #make new centroids
            centroids = tmpCluster
        self.protos = centroids
        #print(self.protos)

    def pickDatapoints(self, numOfNeurons):
    	#print(len(self.scaledData[:]))
    	group = np.random.randint(0, len(self.scaledData[:]), size=numOfNeurons)
    	self.protos = np.vstack([self.protos, self.scaledData[group,:]])
    	self.protos = np.delete(self.protos, len(self.protos[0])-1, axis=1)
    	return self.protos

    def sigma(self, numOfNeurons):
        dTemp = 0
        for i in range(0, numOfNeurons):
            for k in range(0, numOfNeurons):
                dist = np.square(euclidean_distance(self.protos[i], self.protos[k]))
                if dist > dTemp:
                    dTemp = dist
            self.spread[0, i] = dTemp/np.sqrt(numOfNeurons * 2)
        maxv = self.spread.max()
        #set every spread to be max spread
        self.spread[:] = maxv
        #print (self.spread)

    def std_dev(self, numOfNeurons):
        self.spread = np.zeros(shape=(1, numOfNeurons))
        if numOfNeurons == 2:
            self.sigma(numOfNeurons)
        else:
            p = 2
            distances = np.zeros(shape=(numOfNeurons, numOfNeurons))
            neigbours = np.zeros(shape=(numOfNeurons, 2))
            arr = np.zeros(shape=(0, 4))
            for index, centroid in enumerate(self.protos):
                for x in range(0, len(self.protos)):
                    distances[index, x] = np.square(euclidean_distance(self.protos[index], self.protos[x]))
            #print(distances)
            for index, value in enumerate(distances):
                arr = sorted(distances[index, :])[0:4]
                #print(arr)
                if (arr[0] == 0):
                    arr = np.delete(arr, arr[0])
                if (arr[0] == 0):
                    arr = np.delete(arr, arr[0])
                neigbours[index] = arr[0:2]
            #print(neigbours)
            for i in range(0, numOfNeurons):
                self.spread[0, i] = (1/p) * np.power((neigbours[i, 0] + neigbours[i, 1]), 1/2)
            #print("Spread: ", self.spread)

    def train(self, numOfNeurons):
        #seting matrices to zero
        self.protos = np.zeros(shape=(0, len(self.scaledData[1])))
        self.weights = 0
        self.spread = np.zeros(shape=(1, numOfNeurons))

        #self.pickDatapoints(numOfNeurons)
        self.kmeans(numOfNeurons, self.scaledDatanoClassColumn)
        self.sigma(numOfNeurons)
        #self.std_dev(numOfNeurons)

        hiddenOut = np.zeros(shape=(0, numOfNeurons))
        for item in self.scaledDatanoClassColumn:
            out=[]
            for i, proto in enumerate(self.protos):
                distance = np.square(euclidean_distance(item, proto))
                #divide by zero
                if (np.square(self.spread[0, i]) == 0):
                	self.spread[0, i] += 0.000001
                	#print(np.square(self.spread[0, i]))
                neuronOut = np.exp(-(distance)/(2 * np.square(self.spread[0, i])))
                out.append(neuronOut)
            hiddenOut = np.vstack([hiddenOut,np.array(out)])
        #print ("hiddenOut:\n", hiddenOut)

        #print ("klase:\n", self.ClassLabels)
        #print ("pseudo inverz:\n", pinv(hiddenOut).shape)
        if hiddenOut.any():
            self.weights = np.dot(pinv(hiddenOut), self.ClassLabels)
        #print (self.weights)
        #print (self.weights.shape)

    def hiddenSize(self):
    	MSEepoch = np.zeros(shape=(0,0))
    	for size in range(2, 21):
            MSE = np.zeros(shape=(0,0))
            print("Epoch: ", size-2)
            self.train(size)
            #validationData, otherData = train_test_split(self.scaledData, test_size=0.5)
            kfold = KFold(10, True, 5)

            
            for train, test in kfold.split(self.scaledData):
                netOutputs = np.zeros(shape=(0, self.numClasses))
                for item in self.scaledData[train][:,0:-1]:
                    out = []
                    #print(self.protos)
                    for i, proto in enumerate(self.protos):
                        #print(self.scaledData[train][:,0:-1].shape)
                        distance = np.square(euclidean_distance(item, proto))
                        neuronOut = np.exp(-(distance)/(2 * np.square(self.spread[0, i])))
                        out.append(neuronOut)
                    netOut = np.dot(np.array(out),self.weights)
                    #print ("NeuOut:", netOut)
                    netOutputs = np.vstack([netOutputs, netOut])
                #print(netOutputs.shape)
                #exit()

                x, y = self.scaledData[train].shape
                validationDataExact = np.zeros(shape = (0,len(np.unique(self.labels))))
                for i in range(0,x):
                    posNum = self.scaledData[train][i,-1].astype(np.int)
                    newRow = np.zeros(shape=(1, len(np.unique(self.labels))))
                    newRow[:, posNum-1] = 1
                    validationDataExact = np.vstack([validationDataExact, newRow]).astype(np.int)

                #MSE for outputs
                squaredError = 0
                #print(validationDataExact[:5])
                #print(netOutputs[:5])
                for i, value in enumerate(validationDataExact):       	
                	for x in range(0, self.numClasses):
                		squaredError += (validationDataExact[i, x] - netOutputs[i, x]) ** 2
                MSE = np.append(MSE, (squaredError / (len(validationDataExact[:]) * self.numClasses)))
                #print(MSE)
            #print ("Mean squared error:", MSE)
            MSEepoch = np.append(MSEepoch, np.mean(MSE))
            #print(MSEepoch)

    	#plot MSE graph
    	ks = np.arange(2, 21, 1)
    	plt.plot(ks, MSEepoch)
    	plt.xlabel('Number of RBFs')
    	plt.ylabel('MSE')
    	minx = np.argmin(MSEepoch)
    	plt.annotate('min', xy=(ks[minx], np.amin(MSEepoch)), arrowprops=dict(facecolor='black', shrink=0.05))
    	plt.show()
    	self.OptimalCenters = ks[minx]
    	print(self.OptimalCenters)

    def test(self):
        counter = 0
        correct_predictions = 0
        self.accuracy = 0
        self.f1score = 0
        self.precision = 0

        for item in self.testDatanoClassColumn:
            out = []
            for i, proto in enumerate(self.protos):
                distance = np.square(euclidean_distance(item, proto))
                neuronOut = np.exp(-(distance)/(2 * np.square(self.spread[0, i])))
                out.append(neuronOut)
            netOut = np.dot(np.array(out),self.weights)
            #print ('---------------------------------')
            #print (netOut)
            #exit()
            self.PredictionValue = np.vstack([self.PredictionValue, netOut])
            #print(self.PredictionValue.shape)
            #print (np.max(netOut))
            #print (self.threeTestLabels)
            predictedClass = netOut.argmax(axis=0) + 1
            realClass = self.Test_Labels[counter].argmax(axis=0) + 1
            #print ('Predicted class is ',predictedClass)
            #print ('Real class is ', realClass)
            #count the number of correct predictions
            if (predictedClass == realClass):
            	correct_predictions = correct_predictions + 1
            #update predicted array for confusion matrix calculations
            #if self.numClasses == 2:
            #	predictedClass -= 1
            self.predictedLabels = np.append(self.predictedLabels, predictedClass)
            counter = counter + 1

        #calculate accuracy of the network
        #self.MeanSquaredError()
        self.conf_matrix()
        self.ClassificationReport()

        #printing results to txt file
        #sys.stdout = open("experiment_data.csv","w+")
        #print ("Accuracy score: ", accuracy, "%")
        ##self.MeanSquaredError()
        #print (self.conf_matrix)
        #self.ClassificationReport()
        
    #confusion matrix
    def conf_matrix(self):
        for item in self.testLabels:
            self.testLabelsLine = np.append(self.testLabelsLine, item)
        #print ("True: ", self.testLabelsLine)
        #print ("Predicted: ", self.predictedLabels)

        self.conf_matrix = confusion_matrix(self.testLabelsLine, self.predictedLabels)
        #print ("Confusion matrix:")
        #print (self.conf_matrix)

    def MeanSquaredError(self):
    	squaredError = 0
    	for i, value in enumerate(self.Test_Labels):       	
            for x in range(0, self.numClasses):
                squaredError += (self.Test_Labels[i, x] - self.PredictionValue[i, x]) ** 2
    	self.MSE = squaredError / (len(self.PredictionValue[:]) * self.numClasses)
    	print ("Mean squared error:", self.MSE)

    def ClassificationReport(self):
        report = classification_report(self.testLabelsLine, self.predictedLabels)
        print (report)
        self.f1score = f1_score(self.testLabelsLine, self.predictedLabels, average='weighted')
        self.precision = precision_score(self.testLabelsLine, self.predictedLabels, average='weighted')
        self.accuracy = accuracy_score(self.testLabelsLine, self.predictedLabels)



###TESTING RBFN###
Total_conf_matrix = np.zeros(shape=(len(np.unique(dataset[:,-1])),len(np.unique(dataset[:,-1]))))

#FIND OPTIMAL NUMBER OF HIDDEN NEURONS
#x,y = dataset.shape
##determine test size from dataset size
#if (x<500): test_size = 0.4
#elif (x>=500 and x<1000): test_size = 0.8
#elif (x>=1000 and x<10000): test_size = 0.9
#elif (x>=10000): test_size = 0.99
##stratify split to have equal class distribution
#class_label = dataset[:,-1]
#data_noclass = dataset[:, :-1]
#stratSplit = StratifiedShuffleSplit(class_label, 1, test_size=test_size,random_state=42)
#StratifiedShuffleSplit(class_label, n_iter=1, test_size=0.5)
#for train_idx,test_idx in stratSplit:
#    X_train=data_noclass[train_idx]
#    y_train=class_label[train_idx]
#
#train = np.concatenate([X_train, y_train[:, np.newaxis]], axis=1)
#test = dataset[1:5,:]
##train, test = train_test_split(dataset, test_size=test_size)
#networkTest = RBFNetwork(train, test, numClasses)
#networkTest.hiddenSize()
#exit()


##NORMAL SINGULAR TRAIN AND TEST
#split data into k-folds for cross validation
#kfold = KFold(10, True, None)
train, test = train_test_split(dataset, test_size=0.3)
#intialize arrays for classification metrics
Acc = np.zeros(shape=(0,0))
F1 = np.zeros(shape=(0,0))
Precision = np.zeros(shape=(0,0))
#train and test network k-fold times
#for train, test in kfold.split(dataset):
    #network = RBFNetwork(dataset[train], dataset[test], numClasses, NumOfRBF=networkTest.OptimalCenters)
network = RBFNetwork(train, test, numClasses, NumOfRBF=20)
network.train(network.RBFs)
network.test()
    #calculate mean accuracy and standard deviation
    #print(network.conf_matrix.shape)
    #print(Total_conf_matrix.shape)
    #exit()
    #if (network.conf_matrix.shape != Total_conf_matrix.shape):
    #    print("Total confusion matrix will not be updated because of shape irregularity.")
    #if (network.conf_matrix.shape == Total_conf_matrix.shape):
    #    Total_conf_matrix = np.add(Total_conf_matrix, network.conf_matrix)
Acc = np.append(Acc, network.accuracy)
F1 = np.append(F1, network.f1score)
Precision = np.append(Precision, network.precision)
print(F1)
print(Precision)
#print("Average accuracy is: ", np.mean(Acc))
#print("Standard deviation is: ", Acc.std(ddof=1))
#accuracycalc = 100 * np.mean(Acc)
#deviation = 100 * Acc.std()
#print("Global confusion matrix: \n", Total_conf_matrix)
#print("Accuracy: %0.3f (+/- %0.3f)" % (accuracycalc, deviation))
#print("F1 score: %0.5f (+/- %0.5f)" % (np.mean(F1), F1.std()))
#print("Precision: %0.5f (+/- %0.5f)" % (np.mean(Precision), Precision.std()))
exit()


#MAKE DATA FOR TABLES
for i in range(2,21):
    #split data into k-folds for cross validation
    kfold = KFold(10, True, None)
    #intialize arrays for classification metrics
    Acc = np.zeros(shape=(0,0))
    F1 = np.zeros(shape=(0,0))
    Precision = np.zeros(shape=(0,0))
    
    #train and test network k-fold times
    for train, test in kfold.split(dataset):
        #network = RBFNetwork(dataset[train], dataset[test], numClasses, NumOfRBF=networkTest.OptimalCenters)
        network = RBFNetwork(dataset[train], dataset[test], numClasses, NumOfRBF=i)
        network.train(i)
        network.test()
        #calculate mean accuracy and standard deviation
        #Total_conf_matrix = np.add(Total_conf_matrix, network.conf_matrix)
        Acc = np.append(Acc, network.accuracy)
        F1 = np.append(F1, network.f1score)
        Precision = np.append(Precision, network.precision)
    #print(Acc)
    #print(F1)
    #print("Average accuracy is: ", np.mean(Acc))
    #print("Standard deviation is: ", Acc.std(ddof=1))
    accuracycalc = 100 * np.mean(Acc)
    deviation = 100 * Acc.std()
    #print("Global confusion matrix: \n", Total_conf_matrix)
    print("Accuracy: %0.3f (+/- %0.3f)" % (accuracycalc, deviation))
    print("F1 score: %0.5f (+/- %0.5f)" % (np.mean(F1), F1.std()))
    print("Precision: %0.5f (+/- %0.5f)" % (np.mean(Precision), Precision.std()))

    #printing results to txt file
    #sys.stdout = open("experiment_data.csv","w+")
    #print ("Accuracy: %0.3f (+/- %0.3f), " % (100 * np.mean(Acc), 100 * Acc.std()))

    with open("F1_data.csv", "a", newline='') as csvfile:
        fieldnames = ["F1", "deviation"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({"F1": round(np.mean(F1),4), "deviation": round(F1.std(), 4)})

    with open("precision_data.csv", "a", newline='') as csvfile:
        fieldnames = ["precision", "deviation"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({"precision": round(np.mean(Precision),4), "deviation": round(Precision.std(), 4)})
