import numpy as np
from sklearn.svm import LinearSVC

# You are allowed to import any submodules of sklearn as well e.g. sklearn.svm etc
# You are not allowed to use other libraries such as scipy, keras, tensorflow etc

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py
# DO NOT INCLUDE OTHER PACKAGES LIKE SCIPY, KERAS ETC IN YOUR CODE
# THE USE OF ANY MACHINE LEARNING LIBRARIES OTHER THAN SKLEARN WILL RESULT IN A STRAIGHT ZERO

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_predict etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length


def createFeatureVector(x):
    a = np.zeros(shape=(x.shape[0], 16*65+2))
    for i in range(x.shape[0]):
        a[i][-2] = 1
        p = int((2**np.arange(3, -1, -1) * x[i, 64:68]).sum())
        q = int((2**np.arange(3, -1, -1) * x[i, 68:72]).sum())
        a[i][p*65:p*65+64] = -x[i][0:64]
        a[i][q*65:q*65+64] = x[i][0:64]
        a[i][p*65+64] = -1
        a[i][q*65+64] = 1
        a[i][-1] = x[i][-1]
    return a


################################
# Non Editable Region Starting #
################################
def my_fit(Z_train):
    ################################
    #  Non Editable Region Ending  #
    ################################

    train_dt = createFeatureVector(Z_train)
    model = LinearSVC(loss="squared_hinge", max_iter=1000, tol=1e-3)

    train_x = train_dt[:, :-1]
    train_y = train_dt[:, -1]

    # test_x = test_dt[:,:-1]
    # test_y = test_dt[:,-1]

    model.fit(train_x, train_y)
    # Use this method to train your model using training CRPs
    # The first 64 columns contain the config bits
    # The next 4 columns contain the select bits for the first mux
    # The next 4 columns contain the select bits for the second mux
    # The first 64 + 4 + 4 = 72 columns constitute the challenge
    # The last column contains the response

    return model					# Return the trained model


################################
# Non Editable Region Starting #
################################
def my_predict(X_tst, model):
    ################################
    #  Non Editable Region Ending  #
    ################################
    test_dt = createFeatureVector(X_tst)
    test_x = test_dt[:, :-1]
    # Use this method to make predictions on test challenges

    return model.predict(test_x)
