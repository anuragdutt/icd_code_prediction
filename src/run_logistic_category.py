import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
	

if __name__ == "__main__":

	types = np.load("../data/icd_train.types", allow_pickle = True)

	cat = pd.read_excel("../data/ccs_icd_9_and_10_mapping_filtered_for_icd9_and_diabetes.xlsx")
	print(cat.shape)
	cat = cat[cat.code_version == 'ICD9CM']
	print(cat.shape)
	# cat['merged_description'] = cat['code_description']

	icd_heart_failure = cat[cat['ccs_description'].isin(['Congestive heart failure; nonhypertensive'])].loc[:, ['ccs_description', 'code']]
	icd_diabetes = cat[cat['ccs_description'].isin(['Diabetes mellitus without complication', 'Diabetes mellitus with complications'])].loc[:, ['ccs_description','code']]
	icd_chronic_kidney = cat[cat['ccs_description'].isin(['Chronic kidney disease'])].loc[:, ['ccs_description', 'code']]
	icd_acute_kidney = cat[cat['ccs_description'].isin(['Acute and unspecified renal failure'])].loc[:, ['ccs_description', 'code']]
	icd_lupus = cat[cat['code_alternative'].isin(['710.0'])].loc[:, ['ccs_description', 'code']]
	# print(icd_heart_failure)
	# print(icd_diabetes)
	# print(icd_chronic_kidney)
	# print(icd_acute_kidney)
	# print(icd_lupus)


	dat_list = [icd_heart_failure, icd_diabetes, icd_chronic_kidney, icd_acute_kidney, icd_lupus]

	## loading the dataset labevents and icd codes

	lab_train_matrix = np.load("../data/lab_train.matrix", allow_pickle = True)
	lab_train_header = list(np.load("../data/lab_train.types", allow_pickle = True).keys())
	lab_train_pids = np.load("../data/lab_train.pids", allow_pickle = True)

	lab_test_matrix = np.load("../data/lab_test.matrix", allow_pickle = True)
	lab_test_header = list(np.load("../data/lab_test.types", allow_pickle = True).keys())
	lab_test_pids = np.load("../data/lab_test.pids", allow_pickle = True)

	icd_train_matrix = np.load("../data/icd_train.matrix", allow_pickle = True)
	icd_train_header = list(np.load("../data/icd_train.types", allow_pickle = True).keys())
	icd_train_pids = np.load("../data/icd_train.pids", allow_pickle = True)

	icd_test_matrix = np.load("../data/icd_test.matrix", allow_pickle = True)
	icd_test_header = list(np.load("../data/icd_test.types", allow_pickle = True).keys())
	icd_test_pids = np.load("../data/icd_test.pids", allow_pickle = True)

	ret_list = []

	for e in dat_list:
		e['col_code'] = e['code'].apply(lambda x: ''.join(['D_', x]))
		ccs = e['ccs_description'].unique()[0]
		x_train_all = pd.DataFrame(data = icd_train_matrix, index = icd_train_pids, columns = icd_train_header)
		x_test_all = pd.DataFrame(data = icd_test_matrix, index = icd_test_pids, columns = icd_test_header)
		x_train = x_train_all.loc[:, e['col_code'].tolist()]
		x_test = x_test_all.loc[:, e['col_code'].tolist()]
		x_train['sum'] = x_train.sum(axis = 1)
		y_train = x_train['sum'].apply(lambda x: 1 if x > 0 else 0)
		x_test['sum'] = x_test.sum(axis = 1)
		y_test = x_test['sum'].apply(lambda x: 1 if x > 0 else 0)
		x_train = x_train.drop(['sum'], axis = 1)
		x_test = x_test.drop(['sum'], axis = 1)

		lr = LogisticRegression(max_iter = 500, random_state = 0)
		# print("Starting training")
		try:
			lr.fit(x_train, y_train)
			# print("Ending Training")
			y_pred = lr.predict(x_test)
			f1 = f1_score(y_test, y_pred)
			acc = accuracy_score(y_test, y_pred)
			recall = recall_score(y_test, y_pred)
			prec = precision_score(y_test, y_pred)

			prob_true = sum(y_test)/len(y_test)
			prob_pred = sum(y_pred)/len(y_pred)

		except:
			f1 = 0.0
			acc = 0.0
			recall = 0.0
			prec = 0.0
			print("value error detected. one class of values encountered.")

		print(f1)


		retl = [ccs, f1, acc, recall, prec]
		ret_list.append(retl)


		retdf = pd.DataFrame(ret_list, columns = ['ccs_description', "f1", "accuracy", "recall", "precision"])
		retdf.to_csv("../results/logistic_regression.csv", index = False)