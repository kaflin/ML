import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error

diabetes=datasets.load_diabetes()
# print(diabetes.keys()) #It tells what is there in diabetes
# # dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename']) It is basically diabetes keys
# print(diabetes.DESCR)
# print(diabetes.feature_names)
# print(diabetes.target_filename)

# diabetes_X =diabetes.data[:,np.newaxis,2]
diabetes_X =diabetes.data
# print(diabetes_X)

#Features
diabetes_X_train = diabetes_X[:-20] #Last 30 data(-30)
diabetes_X_test =diabetes_X[-20:] #First 20data(-20)

#Labels
diabetes_Y_train = diabetes.target[:-20]
diabetes_Y_test =diabetes.target[-20:]

model =linear_model.LinearRegression()

#To fit our data now
model.fit(diabetes_X_train,diabetes_Y_train)
#Training means,Data are learning to find line or making our data to learn

diabetes_y_predicted=model.predict(diabetes_X_test)
print("Mean squared error is:",mean_squared_error(diabetes_Y_test,diabetes_y_predicted))

print("Weights:",model.coef_)
print("Intercept:",model.intercept_)

plt.scatter(diabetes_X_test,diabetes_Y_test)
plt.plot(diabetes_X_test,diabetes_y_predicted)
plt.show()






