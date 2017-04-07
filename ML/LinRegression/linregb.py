import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
'''
This program is for locally weighted linear regression with data from auto-pilots
we will use first order polynomial to model the feature space for this problem
by Yi Yang
'''
A = np.loadtxt("train_graphs_f16_autopilot_cruise.csv",delimiter=',',skiprows=1)
B = np.loadtxt("test_locreg_f16_autopilot_cruise.csv",delimiter=',',skiprows=1)
rownum = A.shape[0]
colnum = B.shape[1]
ACoeff = A[:,1:colnum-1]
ATarget = A[:,[-1]]
BCoeff = B[:,1:B.shape[1]-1]
BTarget = B[:,[-1]]
# note to attach a unit column vector as the intercept 
#ACoeff = np.insert(ACoeff,0,1,axis=1)
#BCoeff = np.insert(BCoeff,0,1,axis=1)
# will have no use for this problem
'''
def ComputeFeature(degree, InMatrix):
    OutMatrix = InMatrix
    OutMatrix = np.insert(InMatrix,0,1,axis=1)
    newCol = np.empty([InMatrix.shape[0],1])
    for power in range(2,degree+1):
        for j in range(0,InMatrix.shape[1]):
            newCol = np.power(InMatrix[:,j],power)
            OutMatrix = np.insert(OutMatrix,OutMatrix.shape[1],newCol,axis=1)
    return OutMatrix
'''
tau = np.logspace(-2,1,num=10,base=2.0) # set the hyperparameter
# construct locally werighted R matrix
def ConstructR(TrainPoints, PredictPoint,t):
    numPoints = TrainPoints.shape[0]
    Result = np.zeros((numPoints,numPoints))
    for k in range(0,numPoints):
        r = np.exp(-np.power(np.linalg.norm(TrainPoints[k,:]-PredictPoint,2),2)/(2*t**2))
        Result[k,k] = r
    return Result
# compute root mean squared error for test case
predTestTarget = np.empty([BTarget.shape[0],1])
rmseTest = np.empty([tau.shape[0],1])
for index in range(0,tau.shape[0]):
    for i in range(0,BCoeff.shape[0]):
        R = ConstructR(ACoeff,BCoeff[i,:],tau[index])
        '''
        temp = np.dot(np.sqrt(R),ACoeff)
        temp1 = np.dot(np.sqrt(R),ATarget)
        w = np.dot(np.linalg.pinv(temp),temp1)
        '''
        A_T = ACoeff.transpose()
        AA = np.dot(np.dot(A_T,R),ACoeff)
        AA_inv = np.linalg.pinv(AA)
        rt = np.dot(R,ATarget)
        w = np.dot(np.dot(AA_inv,A_T),rt)
        predTestTarget[i][0] = np.dot(BCoeff[[i],:],w)
    rmseTest[index][0] = np.sqrt(mean_squared_error(predTestTarget,BTarget))
# visulize results
plt.plot(tau,rmseTest,'r',label="rmseTest")
plt.xlabel("hyperparameter")
plt.ylabel("RMSE")
plt.legend(loc="upper left",frameon=False)
plt.show()









