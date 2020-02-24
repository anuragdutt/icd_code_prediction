import os
import sys
import numpy as np
import pandas as pd
import _pickle as pickle

if __name__ == "__main__":

	out_l = "../data/labevent"
	out_i = "../data/icd_codes"

	pid_l = np.load("../data/labevent.pids", allow_pickle = True)
	pid_i = np.load("../data/icd_codes.pids", allow_pickle = True)



	pid_common = list(set(pid_l) & set(pid_i))
	pid_i_bool = []
	pid_l_bool = []

	for e in pid_i:
		if e in pid_common:
			pid_i_bool.append(True)
		else:
			pid_i_bool.append(False)


	for e in pid_l:
		if e in pid_common:
			pid_l_bool.append(True)
		else:
			pid_l_bool.append(False)

	inx_l = np.where(np.array(pid_l_bool))
	inx_i = np.where(np.array(pid_i_bool))

	pid_l_common = np.array(pid_l)[inx_l[0]].tolist()
	pid_i_common = np.array(pid_i)[inx_i[0]].tolist()

	mat_l = np.load("../data/labevent.matrix", allow_pickle = True)
	mat_l_common = mat_l[inx_l[0], :]
	mat_i = np.load("../data/icd_codes.matrix", allow_pickle = True)
	mat_i_common = mat_i[inx_i[0], :]

	pickle.dump(mat_l_common, open(out_l+"_common.matrix", "wb"))
	pickle.dump(pid_l_common, open(out_l+"_common.pids", "wb"))
	pickle.dump(mat_i_common, open(out_i+"_common.matrix", "wb"))
	pickle.dump(pid_i_common, open(out_i+"_common.pids", "wb"))

	print(mat_l_common.shape)
	print(len(pid_l_common))
	print(mat_i_common.shape)
	print(len(pid_i_common))
