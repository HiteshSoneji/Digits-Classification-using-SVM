from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()
print("Dataset Size: ", len(digits.data))

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, train_size= 0.7)
print("Training Dataset Size (90 %): ", len(X_train))
print("Testing Dataset Size (10 %): ", len(X_test))

clf = svm.SVC(gamma=0.00006)
clf.fit(X_train, y_train)

predicted = clf.predict(X_test)
expected = y_test

matches = (predicted == expected)
print("The total correctly Predicted values for Test Size of 10 %", matches.sum(), "  Out of ", len(matches))
accuracy = (matches.sum() / float(len(matches)))*100
print("The % accuracy for the model with test size of 10 % is: ", accuracy)

X_train1, X_test1, y_train1, y_test1 = train_test_split(digits.data, digits.target, train_size= 0.5)
print("Training Dataset Size (90 %): ", len(X_train1))
print("Testing Dataset Size (10 %): ", len(X_test1))

clf1 = svm.SVC(gamma=0.00006)
clf1.fit(X_train1, y_train1)

predicted1 = clf1.predict(X_test1)
expected1 = y_test1

matches1 = (predicted1 == expected1)
print("The total correctly Predicted values for Test Size of 10 %", matches1.sum(), "  Out of ", len(matches1))
accuracy1 = (matches1.sum() / float(len(matches1)))*100
print("The % accuracy for the model with test size of 10 % is: ", accuracy1)

X_train2, X_test2, y_train2, y_test2 = train_test_split(digits.data, digits.target, train_size= 0.9)
print("Training Dataset Size (90 %): ", len(X_train))
print("Testing Dataset Size (10 %): ", len(X_test))

clf2 = svm.SVC(gamma=0.00006)
clf2.fit(X_train2, y_train2)

predicted2 = clf2.predict(X_test2)
expected2 = y_test2

matches2 = (predicted2 == expected2)
print("The total correctly Predicted values for Test Size of 10 %", matches2.sum(), "  Out of ", len(matches2))
accuracy2 = (matches2.sum() / float(len(matches2)))*100
print("The % accuracy for the model with test size of 10 % is: ", accuracy2)