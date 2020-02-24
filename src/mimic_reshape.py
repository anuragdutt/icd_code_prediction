import os
import sys
import numpy as np
import pandas as pd
import _pickle as pickle

def reshapeData(dat, inx, col, val):
	df = dat.pivot(index = inx, columns = col, values = val)
	return df

if __name__ == "__main__":

	# labitem = pd.read_csv("../raw/D_LABITEMS.csv.gz", compression = "gzip")

	di = pd.read_csv("../raw/DIAGNOSES_ICD.csv.gz", compression = "gzip")
	di = di.drop(['ROW_ID'], axis = 1)
	di.to_csv("../raw/DIAGNOSES_ICD.csv", index = False)
	ad = pd.read_csv("../raw/ADMISSIONS.csv.gz", compression = "gzip")
	ad = ad.drop('ROW_ID', axis = 1)
	ad.to_csv("../raw/ADMISSIONS.csv", index = False)


	outFile = "../data/labevent"
	le = pd.read_csv("../raw/LABEVENTS.csv.gz", compression = "gzip")
	le.to_csv("../raw/LABEVENTS.csv", index = False)
	le = le.dropna(subset = ["HADM_ID"])
	le_agg = le.groupby(['HADM_ID', 'ITEMID'], as_index = False)['VALUENUM'].median()
	le_dat = reshapeData(le_agg, 'HADM_ID', 'ITEMID', 'VALUENUM')
	print(le_dat.shape)	
	le_matrix = np.asarray(le_dat)

	print(le_matrix.shape)
	# le_median = le_matrix.median(axis = 0)

	pids = le_dat.index.tolist()
	types = dict(zip(le_dat.columns.tolist(), range(len(le_dat.columns.tolist()))))


	pickle.dump(pids, open(outFile+'.pids', 'wb'), -1)
	pickle.dump(le_matrix, open(outFile+'.matrix', 'wb'), -1)
	pickle.dump(types, open(outFile+'.types', 'wb'), -1)




