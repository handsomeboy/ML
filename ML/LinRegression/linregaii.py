import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
'''
Fit the data by using psudo inverse approach of regularized linear regression
by Yi Yang
'''
Coeff = np.loadtxt("train_graphs_f16_autopilot_cruise.csv",delimiter=',',skiprows=1)
rownum = Coeff.shape[0]
colnum = Coeff.shape[1]
TrainCoeff = Coeff[:,1:colnum-1]
TrainTarget = Coeff[:,[-1]]
Coeff1 = np.loadtxt("test_graphs_f16_autopilot_cruise.csv",delimiter=',',skiprows=1)
TestCoeff = Coeff1[:,1:Coeff1.shape[1]-1]
TestTarget = Coeff1[:,[-1]]
def ComputeFeature(degree, InMatrix):
    OutMatrix = InMatrix # output InMatrix if degree equals one
    OutMatrix = np.insert(InMatrix,0,1,axis=1) # insert one unit column
    newCol = np.empty([InMatrix.shape[0],1])
    for power in range(2,degree+1):
        for j in range(0,InMatrix.shape[1]):
            newCol = np.power(InMatrix[:,j],power)
            OutMatrix = np.insert(OutMatrix,OutMatrix.shape[1],newCol,axis=1)
            # the feature matrix is formed as [x1,x2,...,x6,x1^2,x2^2,...]
    return OutMatrix
# compute model coefficients using least square methods
rmseTrain = np.empty([np.arange(-40,21).shape[0],1])
rmseTest = np.empty([np.arange(-40,21).shape[0],1])
for M in range(6,7):
    TrainFeature = ComputeFeature(M,TrainCoeff)
    TestFeature = ComputeFeature(M,TestCoeff)
    TrainFeature_i = TrainFeature.transpose()
    TrainFeature2 = np.dot(TrainFeature_i,TrainFeature)
    i = 0
    for lambda_ln in range(-40,21):
        lambda_ = np.exp(lambda_ln)
        TrainFeature12 = lambda_ * np.eye(TrainFeature2.shape[0],TrainFeature2.shape[1])
        TrainFeature22 = np.add(TrainFeature12,TrainFeature2)
        TrainFeature22_i = np.linalg.inv(TrainFeature22)
        temp = np.dot(TrainFeature_i,TrainTarget)
        w = np.dot(TrainFeature22_i,temp)
        TrainFix = np.dot(TrainFeature,w)
        TestFix = np.dot(TestFeature,w)
        rmseTrain[i][0] = np.sqrt(mean_squared_error(TrainFix,TrainTarget))
        rmseTest[i][0] = np.sqrt(mean_squared_error(TestFix,TestTarget))
        i += 1

# plot training error and test error as RMSE against M
lambda_ln = range(-40,21)
plt.plot(lambda_ln,rmseTrain,color="red",linestyle="-",label="rmseTrain")
plt.plot(lambda_ln,rmseTest,color="blue",linestyle="--",label="rmseTest")
plt.xlabel("Regularized hyperparameter")
plt.ylabel("RMSE")
plt.legend(loc="upper left",frameon=False)
plt.show()

