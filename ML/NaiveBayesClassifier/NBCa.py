import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
'''
Specification: Refer to http://www.graphpad.com/support/faqid/1089/, it gives definitions to nominal/ordinal/interval/ratio variables
The preprocessing step cannot be applied to nominal(categorical)feature variables. In spambase, features are word frequencies,
charater frequencies, length of capital letter in a row, and the number of captital letters, none of which are nominal
by Yi Yang
'''
A = np.loadtxt("spambase.train",delimiter=",")







