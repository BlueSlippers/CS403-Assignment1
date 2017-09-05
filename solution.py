import numpy as np
import pandas as pd
import pickle
#from matplotlib import pyplot as plt


def get_feature_matrix(file_path):
    feature_matrix = pd.read_csv(file_path)

    # Get the data, month and hour from single column
    feature_matrix['date'] = pd.to_datetime(feature_matrix.date)
    feature_matrix['month'] = pd.DatetimeIndex(feature_matrix['date']).month
    feature_matrix['hour'] = pd.DatetimeIndex(feature_matrix['date']).hour

    # With the hunch that the time of the day is crucial,
    # We will create additional features representing which
    # hour of the day it is
    feature_matrix['0'] = feature_matrix['hour']==0
    feature_matrix['1'] = feature_matrix['hour']==1
    feature_matrix['2'] = feature_matrix['hour']==2
    feature_matrix['3'] = feature_matrix['hour']==3
    feature_matrix['4'] = feature_matrix['hour']==4
    feature_matrix['5'] = feature_matrix['hour']==5
    feature_matrix['6'] = feature_matrix['hour']==6
    feature_matrix['7'] = feature_matrix['hour']==7
    feature_matrix['8'] = feature_matrix['hour']==8
    feature_matrix['9'] = feature_matrix['hour']==9
    feature_matrix['10'] = feature_matrix['hour']==10
    feature_matrix['11'] = feature_matrix['hour']==11
    feature_matrix['12'] = feature_matrix['hour']==12
    feature_matrix['13'] = feature_matrix['hour']==13
    feature_matrix['14'] = feature_matrix['hour']==14
    feature_matrix['15'] = feature_matrix['hour']==15
    feature_matrix['16'] = feature_matrix['hour']==16
    feature_matrix['17'] = feature_matrix['hour']==17
    feature_matrix['18'] = feature_matrix['hour']==18
    feature_matrix['19'] = feature_matrix['hour']==19
    feature_matrix['20'] = feature_matrix['hour']==20
    feature_matrix['21'] = feature_matrix['hour']==21
    feature_matrix['22'] = feature_matrix['hour']==22
    feature_matrix['23'] = feature_matrix['hour']==23


    # With the assumptions that seasons play a major role
    # in electricity usage
    feature_matrix['Jan'] = feature_matrix['month']==1
    feature_matrix['Feb'] = feature_matrix['month']==2
    feature_matrix['Mar'] = feature_matrix['month']==3
    feature_matrix['Apr'] = feature_matrix['month']==4
    feature_matrix['May'] = feature_matrix['month']==5

    # Additional column for bias
    feature_matrix['ones'] = 1

    # Convert booleans to int 0 or 1
    feature_matrix.iloc[:,:] = feature_matrix.astype(int)

    #In case for testing data
    try:
        return_matrix = feature_matrix.drop(['Output'], inplace=True, axis=1)
    except ValueError:
        pass

    # Not needed as laready represented by other features
    matrix = feature_matrix.drop(['Id', 'date', 'month', 'hour'], axis=1)

    # Scaling to enable the ease of convergence
    matrix = matrix * 1. / np.max(matrix, axis=0)


    return np.asarray(matrix)

def get_output(file_path):
    required_matrix = pd.read_csv(file_path)
    required_output = required_matrix['Output']

    return np.asarray(required_output)


def get_weight_vector(feature_matrix, output, lambda_reg, p):
    # Set initial weights = 1
    weights = np.ones(feature_matrix.shape[1])
    #error_list = []

    #delta = 1

    def regularization_portion(lambda_reg, p_, weights):

        if p_ == 1:
            r_error = np.dot(lambda_reg, np.ones(weights.shape[0]))
            r_error[-1] = 0

            return r_error
        elif p_ == 2:

            r_error = np.dot((lambda_reg / (np.linalg.norm(abs(weights)))), abs(weights))
            r_error[-1] = 0

            return r_error

        else:

            numerator = lambda_reg * ((((abs(weights)) ** p).sum(axis=0)) ** (1 / p))
            denominator = ((abs(weights)) ** p).sum(axis=0)

            r_error = np.dot((numerator / denominator), (abs(weights) ** (p - 1)))
            r_error[-1] = 0

            return r_error

    def sq_error_portion(features, weights, output):

        error = np.dot((np.dot(features, weights) - output), features)
        return error

    #Step size
    alpha = 0.01
   
    for x in range(2):
        for features_, output_ in zip(feature_matrix, output):

            grad = sq_error_portion(features_, weights, output_) + regularization_portion(lambda_reg, p, weights)


            weights = weights - np.dot(alpha, grad)


            RMSE = np.sqrt((abs(np.dot(feature_matrix, weights) - output).sum()) / (feature_matrix.shape[0]))
            #error_list.append(RMSE)

    return weights

def get_my_best_weight_vector():
    '''
    #<-- Saving weights with optimum values -->#
        optimum parameter values are found by
        plotting graphs. DOne later



    train_data = get_feature_matrix("train.csv")
    train_results = get_output("train.csv")

    final_weights = get_weight_vector(train_data, train_results, 0.1, 1.2)

    with open('weights.pkl', 'wb') as f:
   '''
    
   #<--Load Weights from pickle-->$
    with open('weights.pkl', 'rb') as f:
        final_weights = pickle.load(f)

    return final_weights


def save_as_csv():
    predictions = np.dot(get_feature_matrix("test_features.csv"),get_my_best_weight_vector())
    df = pd.DataFrame(predictions)

    df.to_csv("output.csv",  index=True)

#get_my_best_weight_vector()
#save_as_csv()



###################################################################################

'''############################################################
    For plotting Error function for various values of p
    when all others parameters remain constant.

    The graph is saved in th folder
   ############################################################
'''
'''
train_data = get_feature_matrix("train.csv")[:8000]
test_data = get_feature_matrix("train.csv")[8000:]


train_results = get_output("train.csv")[:8000]
test_results = get_output("train.csv")[8000:]


list_x = []
list_y = []
for x in np.arange(1.0,2.0,0.1):
    final_weights, train_errors = get_weight_vector(train_data,train_results,4,x)
    list_y.append(np.sqrt((abs(np.dot(test_data, final_weights) - test_results).sum()) / 5000))
    print list_y[-1]
    list_x.append(x)
    print list_x[-1]
    print "__"
    print float(x)

plt.plot(list_x, list_y)
plt.show()

'''
'''############################################################
    For plotting Error function for various values of lambda
    when all others parameters remain constant.

    The graph is saved in th folder
   ############################################################
'''
'''
list_x = []
list_y = []
for x in np.arange(0.1,4.0,0.1):
    final_weights, train_errors = get_weight_vector(train_data,train_results,x,1.2)
    list_y.append(np.sqrt((abs(np.dot(test_data, final_weights) - test_results).sum()) / 5000))
    print list_y[-1]
    list_x.append(x)
    print list_x[-1]
    print "__"
    print float(x)

plt.plot(list_x, list_y)
plt.show()
'''
########################################################################################
'''
x = []
y = []
z = []

counter = 0
for p in np.arange(1.0,1.5,0.1):
    for lmda in np.arange(0.1,1.0,0.1):
        final_weights, train_errors = get_weight_vector(train_data, train_results, lmda, p)
        z.append(np.sqrt((abs(np.dot(test_data, final_weights) - test_results).sum()) / 5000))
        x.append(p)
        y.append(lmda)
        counter +=1
        print counter

print x
print y
print z


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter



fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.


# Plot the surface.
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-10.01, 10.01)
#ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
'''