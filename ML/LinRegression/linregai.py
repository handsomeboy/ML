import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
'''
multidimensional linear regression using polynomial model
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
rmseTrain = np.empty([6,1])
rmseTest = np.empty([6,1])
for M in range(1,7):
    TrainFeature = ComputeFeature(M,TrainCoeff)
    TestFeature = ComputeFeature(M,TestCoeff)
    Model = np.dot(np.linalg.pinv(TrainFeature),TrainTarget)
    TrainFix = np.dot(TrainFeature,Model)
    TestFix = np.dot(TestFeature,Model)
    rmseTrain[M-1][0] = np.sqrt(mean_squared_error(TrainFix,TrainTarget))
    rmseTest[M-1][0] = np.sqrt(mean_squared_error(TestFix,TestTarget))

# plot training error and test error as RMSE against M
Mdegree = range(1,7)
plt.plot(Mdegree,rmseTrain,color="red",linestyle="-",label="rmseTrain")
plt.plot(Mdegree,rmseTest,color="blue",linestyle="--",label="rmseTest")
plt.xlabel("Order of the polynomial")
plt.ylabel("RMSE")
plt.legend(loc="upper left",frameon=False)
plt.show()









