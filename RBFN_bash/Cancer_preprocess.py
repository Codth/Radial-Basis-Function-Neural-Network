import numpy as np
import csv
from pandas import read_csv
import pandas as pd
from sklearn.model_selection import train_test_split

with open('cancer-data.csv') as f:
    raw_data = f.read()

####PREPROCESS OF THE DATASET######
def data_preprocess(raw_data):
    # Read string files
    dataset = list()
    csv_reader = csv.reader(raw_data.split('\n'), delimiter=',')
    for row in csv_reader:
        if not row:
            continue
        dataset.append(row)
    pd_data = pd.DataFrame(dataset)

    labels = pd_data.iloc[:,-1].values
    labels = labels[:, np.newaxis]
    #CONVERTING TEXT CLASS LABELS TO NUMBERS
    b, c = np.unique(labels, return_inverse=True)
    labels = c[:, np.newaxis] + 1
    labels = pd.DataFrame(labels)
    #delete last column from dataframe
    pd_data.drop(pd_data.columns[len(pd_data.columns)-1], axis=1, inplace=True)
    #concatenate data with numerical class labels
    result = pd.concat([pd_data, labels], axis=1)
    

    #replace question marks with NaN
    result = result.replace(['?', '-'], np.nan)
    # drop rows with missing values
    result = result.dropna()

    dataset = result.values
    dataset = np.array(dataset).astype(np.float)

    # Find the min and max values for each column
    stats = [[min(column), max(column)] for column in zip(*dataset)]

    # Rescale dataset columns to the range 0-1 - normalization
    for row in dataset:
        for i in range(len(row)-1):
            row[i] = (row[i] - stats[i][0]) / (stats[i][1] - stats[i][0])
    return dataset


#def data_preprocess2(filename):
#	# Load a CSV file
#	dataset = read_csv(filename, header=None)
#	# mark zero values as missing or NaN
#	dataset[[1,2,3,4,5,6,7,8,9]] = dataset[[1,2,3,4,5,6,7,8,9]].replace('?', np.NaN)
#	# drop rows with missing values
#	dataset.dropna(inplace=True)
#	dataset = np.array(dataset).astype(float)
#	dataset = np.delete(dataset, 0, axis=1)
#	print (dataset.shape)
#
#	# Find the min and max values for each column
#	stats = [[min(column), max(column)] for column in zip(*dataset)]
#
#	# Rescale dataset columns to the range 0-1 - normalization
#	for row in dataset:
#		for i in range(len(row)-1):
#			row[i] = (row[i] - stats[i][0]) / (stats[i][1] - stats[i][0])
#
#	for i, data in enumerate(dataset):
#		if data[-1] == 2:
#			data[-1] = 0
#		if data[-1] == 4:
#			data[-1] = 1
#
#	dataset = dataset[np.argsort(dataset[:, -1])]
#	return dataset

#np.set_printoptions(threshold=np.inf)

dataset = data_preprocess(raw_data)
dataset = dataset[dataset[:,1].argsort()]
#print(dataset)
#print (dataset.shape)
#count the number of classes
numClasses = len(np.unique(dataset[:,-1]))
#print(numClasses)
#exit()
#train, test = train_test_split(dataset, test_size=0.2)
