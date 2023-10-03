import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the training data
X_train = np.loadtxt('X_train.csv', delimiter=',')
y_train = np.loadtxt('y_train.csv', delimiter=',')

# Report the class frequencies
print('Class frequencies:')
for i in range(5):
    print('Class %d: %f' % (i, np.sum(y_train == i) / y_train.shape[0]))


# Split the training data into training and validation data (80/20 split)
X_train_soft = X_train[:int(0.8 * X_train.shape[0])]
y_train_soft = y_train[:int(0.8 * y_train.shape[0])]
X_val_soft = X_train[int(0.2 * X_train.shape[0]):]
y_val_soft = y_train[int(0.2 * y_train.shape[0]):]


# Train softmax regression model
softmax_model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
softmax_model.fit(X_train_soft, y_train_soft)

# Make predictions on the validation data
y_pred = softmax_model.predict(X_val_soft)

# Report the accuracy
print('Accuracy: ', accuracy_score(y_val_soft, y_pred))