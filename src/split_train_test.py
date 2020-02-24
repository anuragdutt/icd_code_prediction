import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import _pickle as pickle

if __name__ == "__main__":

	lab_mat = np.load("../data/labevent_common.matrix", allow_pickle = True)
	icd_mat = np.load("../data/icd_codes_common.matrix", allow_pickle = True)

	lab_index = np.load("../data/labevent_common.pids", allow_pickle = True)
	icd_index = np.load("../data/icd_codes_common.pids", allow_pickle = True)

	lab_header_dict = np.load("../data/labevent.types", allow_pickle = True)
	lab_header = list(lab_header_dict.keys())
	icd_header_dict = np.load("../data/icd_codes.types", allow_pickle = True)
	icd_header = list(icd_header_dict.keys())


	lab = pd.DataFrame(data = lab_mat, index = lab_index, columns = lab_header)
	icd = pd.DataFrame(data = icd_mat, index = icd_index, columns = icd_header)


	lab = lab.sort_index(axis = 0)
	icd = icd.sort_index(axis = 0)



	## Splitting Icd dataset at 80-20
	y_range = list(range(icd.shape[0]))
	lab_median = lab.median(axis = 0)
	nan_header = np.array(lab_header)[lab_median.isna()].tolist()
	lab = lab.drop(nan_header, axis = 1)
	lab_train_header = lab.columns.tolist()

	lab_train, lab_test, icd_train, icd_test = train_test_split(lab, icd, test_size = 0.2, random_state = 16) 
	
	print(lab_train.shape)
	print(lab_test.shape)
	print(icd_train.shape)
	print(icd_test.shape)

	lab_train_median = lab_train.median(axis = 0)
	lab_train = lab_train.fillna(lab_train_median)
	nan_train_header = np.array(lab_train_header)[lab_train_median.isna()].tolist()
	lab_train = lab_train.drop(nan_train_header, axis = 1)
	lab_test = lab_test.drop(nan_train_header, axis = 1)
	lab_train_header = lab_train.columns.tolist()
	lab_test_header = lab_train_header

	print(len(lab_train_header))
	print(len(lab_header))

	lab_train_matrix = np.asarray(lab_train)
	lab_train_header = dict(zip(lab_train_header, range(len(lab_train_header))))
	lab_train_pids = lab_train.index.tolist()

	lab_test_matrix = np.asarray(lab_test)
	lab_test_header = dict(zip(lab_test_header, range(len(lab_test_header))))
	lab_test_pids = lab_test.index.tolist()

	icd_train_matrix = np.asarray(icd_train)
	icd_train_header = dict(zip(icd_header, range(len(icd_header))))
	icd_train_pids = icd_train.index.tolist()

	icd_test_matrix = np.asarray(icd_test)
	icd_test_header = dict(zip(icd_header, range(len(icd_header))))
	icd_test_pids = icd_test.index.tolist()


	pickle.dump(lab_train_matrix, open("../data/lab_train.matrix", "wb"))
	pickle.dump(lab_train_pids, open("../data/lab_train.pids", "wb"))
	pickle.dump(lab_train_header, open("../data/lab_train.types", "wb"))

	pickle.dump(lab_test_matrix, open("../data/lab_test.matrix", "wb"))
	pickle.dump(lab_test_pids, open("../data/lab_test.pids", "wb"))
	pickle.dump(lab_test_header, open("../data/lab_test.types", "wb"))

	pickle.dump(icd_train_matrix, open("../data/icd_train.matrix", "wb"))
	pickle.dump(icd_train_pids, open("../data/icd_train.pids", "wb"))
	pickle.dump(icd_train_header, open("../data/icd_train.types", "wb"))

	pickle.dump(icd_test_matrix, open("../data/icd_test.matrix", "wb"))
	pickle.dump(icd_test_pids, open("../data/icd_test.pids", "wb"))
	pickle.dump(icd_test_header, open("../data/icd_test.types", "wb"))

