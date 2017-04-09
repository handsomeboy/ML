import numpy as np
from matplotlib import pyplot as plt
import sys
import csv
'''
Task1: ridge regression, output w based on given lambda
Task2: active learning, output the first 10 locations from test sets 
based on given trained sets and an arbitray setting of {lambda,sigma2}
by Yi Yang
'''
np.set_printoptions(precision=2)
# Part 1
lambda_ = float(sys.argv[1])
sigma2 = float(sys.argv[2])
X_train = sys.argv[3]
y_train = sys.argv[4]
X_test = sys.argv[5]
TrainCoeff = np.loadtxt(X_train,delimiter=',',skiprows=0)
TrainTarget = np.loadtxt(y_train,delimiter=',',skiprows=0)
TestCoeff = np.loadtxt(X_test,delimiter=',',skiprows=0)
I = np.eye(TrainCoeff.shape[1])
w_rr = np.dot(np.dot(np.linalg.inv(lambda_*I + np.dot(TrainCoeff.T,TrainCoeff)),TrainCoeff.T),TrainTarget)
np.savetxt("wRR" + "_" + sys.argv[1] + ".csv", w_rr, delimiter=",")

# Part 2
Sigma_trainPost = np.linalg.inv(lambda_*I + np.dot(TrainCoeff.T,TrainCoeff)/sigma2)
mu_trainPost = np.dot(np.linalg.inv(lambda_*sigma2*I + np.dot(TrainCoeff.T,TrainCoeff)),np.dot(TrainCoeff.T,TrainTarget))
# form the predictive distribution p(y0|x0,y,X) for all unmeasured x0 element of D
mu_testPred = np.dot(TestCoeff,mu_trainPost)
sigma2_testPred = np.zeros((TestCoeff.shape[0],1))
for i in range(TestCoeff.shape[0]):
    sigma2_testPred[i][0] = sigma2 + np.dot(np.dot(TestCoeff[[i],:],Sigma_trainPost),TestCoeff[[i],:].T)
# pick the x0 for which sigma2 is largest, see lecture 5 in ColumbiaX
index = np.argsort(-1 * sigma2_testPred,axis=0)[0:10]
indexList = index.flatten().tolist()
#np.savetxt("active" + "_" + sys.argv[1] + "_" + sys.argv[2] + ".csv",indexList,fmt='%d',delimiter=',')
with open("active" + "_" + sys.argv[1] + "_" + sys.argv[2] + ".csv", 'w') as myfile:
    wr = csv.writer(myfile)
    wr.writerow(indexList)







