import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import GridSearchCV	
from sklearn.ensemble import RandomForestClassifier

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy

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
	print("lab train matrix shape:")
	print(lab_train_matrix.shape)
	print("icd train matrix shape:")
	print(icd_train_matrix.shape)


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

		print("Total sum of cases for each disease:")
		print(np.sum(y_train))
		print("icd codes in each iteration:")
		print(len(e))
		# print("Starting training")

		base_model = RandomForestClassifier(n_estimators = 50, random_state = 0)
		base_model.fit(lab_train_matrix, y_train)
		base_accuracy = evaluate(base_model, lab_test_matrix, y_test)

		param_grid = {
		    'bootstrap': [True],
		    'max_depth': [50,100, 200, 300],
		    'min_samples_leaf': [10, 30, 50, 70, 90],
		    'min_samples_split': [10, 20, 50],
		    'n_estimators': [200, 400, 600, 800, 1000]
		}

		rf = RandomForestClassifier()

		grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
    		                      cv = 3, n_jobs = -1, verbose = 2)
		
		grid_search.fit(lab_train_matrix, y_train)
		best_grid = grid_search.best_estimator_

		print(grid_search.best_params_)

		print("Ending Training")
		grid_accuracy = evaluate(best_grid, lab_test_matrix, y_test)

		y_pred = grid_search.predict(lab_test_matrix)
		f1 = f1_score(y_test, y_pred)
		acc = accuracy_score(y_test, y_pred	)
		recall = recall_score(y_test, y_pred)
		prec = precision_score(y_test, y_pred)

		# except:
		# 	f1 = 0.0
		# 	acc = 0.0
		# 	recall = 0.0
		# 	prec = 0.0
		# 	print("value error detected. one class of values encountered.")

		print("The f1 score of the iteration is:", f1)

		retl = [ccs, f1, acc, recall, prec]
		ret_list.append(retl)


		retdf = pd.DataFrame(ret_list, columns = ['ccs_description', "f1", "accuracy", "recall", "precision"])
		retdf.to_csv("../results/random_forest.csv", index = False)