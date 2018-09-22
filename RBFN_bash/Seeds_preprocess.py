import numpy as np
from csv import reader
from sklearn.model_selection import train_test_split


def data_preprocess(filename):
	# Load a CSV file
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	dataset = np.array(dataset).astype(np.float)

	# Find the min and max values for each column
	stats = [[min(column), max(column)] for column in zip(*dataset)]

	# Rescale dataset columns to the range 0-1 - normalization
	for row in dataset:
		for i in range(len(row)-1):
			row[i] = (row[i] - stats[i][0]) / (stats[i][1] - stats[i][0])
	return dataset


dataset = data_preprocess('seeds_dataset.csv')
#print (dataset)

#count the number of classes
numClasses = len(np.unique(dataset[:,-1]))
#print(numClasses)

#split dataset into train and test group
#train, test = train_test_split(dataset, test_size=0.2)

#split data into k-folds for cross validation
#kfold = KFold(10, True, 1)
#for train, test in kfold.split(dataset):
#	print('train: %s, test: %s' % (dataset[train], dataset[test]))


