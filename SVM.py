
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn import svm

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

x = [2,3.1,1.7,1.9,2.4]
x2 = [3.6,3.7,4.1,3.9,6]


y = [4,6,5,5.2,3.6]
y2 = [4.1,7,6,5.5,5]

plt.scatter(x,y,color='r')     #Set A(red)
plt.scatter(x2,y2,color='b')   #Set B(blue)
plt.show()



x = np.array([[2,4],
             [3.1,6],
             [1.7,5],
             [1.9,5.2],
             [2.4,3.6]])

x2 = np.array([[3.6,4.1],
             [3.7,7],
             [4.1,6],
             [3.9,5.5],
             [6,5]])

z = np.concatenate((x,x2), axis=0)
y = (0,0,0,0,0,1,1,1,1,1)


clf = svm.SVC(kernel='linear', C = 1.0)
clf.fit(z,y)

w = clf.coef_[0]
#print(w)

a = -w[0] / w[1]

xx = np.linspace(0,7)
yy = a * xx - clf.intercept_[0] / w[1]

h0 = plt.plot(xx, yy, 'k-')

plt.scatter(z[:, 0], z[:, 1], c = y)
#plt.legend()
plt.show()

#prediction
print("\nPREDICTION")
print("Enter your coordinates for prediction.,\n")

n1 = input('x:coordinate : ')
n2 = input('y-coordinate : ')

a=  clf.predict([[n1,n2]])
#print (a)

if a == [0]:
       print("\nSet-A")
else:
       print("\nSet-B")

#Accurancy
       
z_train, z_test, y_train, y_test = train_test_split(z, y, test_size=0.38, random_state=1)
clf = svm.SVC(kernel='linear')
clf.fit(z_train, y_train)

print("\nAccuracy Obtained for the above data set : {}".format(clf.score(z_train, y_train)))



