import numpy as np

from scipy.optimize import curve_fit

#oldCoord is an array with the IVM coordinates
#newCoord is an array with the juxta coordinates, that we want to achieve with the transformation

oldCoord=np.genfromtxt(r'E:\Paper\correcao software\oldCoordinatesIVM4.dat', delimiter=',')
newCoord=np.genfromtxt(r'E:\Paper\correcao software\newCoordinates4_Yabs.dat', delimiter=',')
newCoordErrors=np.genfromtxt(r'E:\Paper\correcao software\newcoordinates_errors.dat', delimiter=',')

n_points=np.shape(oldCoord)[0]

#Mechanical calibration

matrix=np.array([(0.745476, 0.0, 0.66653247), (0.0, 1.0, 0.0), (0.66653247, 0.0, -0.745476)])

blah=0
for i in range(n_points):
    blah=(matrix[0][0]*oldCoord[i][0]+matrix[0][1]*oldCoord[i][1]+matrix[0][2]*oldCoord[i][2] - newCoord[i][0])**2
    blah= blah + (matrix[1][0]*oldCoord[i][0]+matrix[1][1]*oldCoord[i][1]+matrix[1][2]*oldCoord[i][2] - newCoord[i][1])**2
    blah = blah + (matrix[2][0]*oldCoord[i][0]+matrix[2][1]*oldCoord[i][1]+matrix[2][2]*oldCoord[i][2] - newCoord[i][2])**2

    print (np.sqrt(blah))

evaluatedCoordinates = np.zeros(np.shape(oldCoord))
evaluatedCoordinates = np.dot(matrix,oldCoord.T)
#the following loop are the transformed coordinates

for i in range(n_points):
    print (evaluatedCoordinates[0,i] , evaluatedCoordinates[1,i] , evaluatedCoordinates[2,i])
    print (newCoord[i][0] , newCoord[i][1] , newCoord[i][2])
    print ('')


# Software calibration
def linearfunc(x, a, b, c ):
    return a*x[0]+b*x[1]+c*x[2]

poptx, pcovx = curve_fit(linearfunc, oldCoord.transpose(), newCoord[:,0],sigma=newCoordErrors)
popty, pcovy = curve_fit(linearfunc, oldCoord.transpose(), newCoord[:,1],sigma=newCoordErrors)
poptz, pcovz = curve_fit(linearfunc, oldCoord.transpose(), newCoord[:,2],sigma=newCoordErrors)

matrix=np.zeros((3,3))

for i in range(3):
    matrix[0][i]=poptx[i]
for i in range(3):
    matrix[1][i]=popty[i]
for i in range(3):
    matrix[2][i]=poptz[i]



# blah is the error calculated as the difference between transformed coordinates and the newCoord elements
blah=0
for i in range(n_points):
    blah=(matrix[0][0]*oldCoord[i][0]+matrix[0][1]*oldCoord[i][1]+matrix[0][2]*oldCoord[i][2] - newCoord[i][0])**2
    blah= blah + (matrix[1][0]*oldCoord[i][0]+matrix[1][1]*oldCoord[i][1]+matrix[1][2]*oldCoord[i][2] - newCoord[i][1])**2
    blah = blah + (matrix[2][0]*oldCoord[i][0]+matrix[2][1]*oldCoord[i][1]+matrix[2][2]*oldCoord[i][2] - newCoord[i][2])**2

    print (np.sqrt(blah))
#    print matrix[0][0]*oldCoord[i][0]+matrix[0][1]*oldCoord[i][1]+matrix[0][2]*oldCoord[i][2] , ", ", matrix[1][0]*oldCoord[i][0]+matrix[1][1]*oldCoord[i][1]+matrix[1][2]*oldCoord[i][2] , ", ", matrix[2][0]*oldCoord[i][0]+matrix[2][1]*oldCoord[i][1]+matrix[2][2]*oldCoord[i][2]

evaluatedCoordinates = np.zeros(np.shape(oldCoord))
evaluatedCoordinates = np.dot(matrix,oldCoord.T)
#the following loop are the transformed coordinates

for i in range(n_points):
    print (evaluatedCoordinates[0,i] , evaluatedCoordinates[1,i] , evaluatedCoordinates[2,i])
    print (newCoord[i][0] , newCoord[i][1] , newCoord[i][2])
    print ('')


#From coordinates after angle correction to pure coordinates

a = np.array([[0.745476,0.66653247],[0.66653247,-0.745476]])
b = np.array([113.3769,	-4417.219]) # coordinates saved by bonsai w angle transform
x = np.linalg.solve(a,b)
print(x)

#From coordinates after software correction to pure coordinates

a = np.array([[ 7.36412650e-02, 1.25686386e-03,6.61695851e-02 ],[-1.68983025e-04,   1.01548734e-01,  -1.14021148e-03],[6.77851468e-02,  -8.28774803e-04,  -7.44689774e-02]])
b = np.array([-83.87363,	97.11988,	-2771.447]) # coordinates saved by bonsai w angle transform
x = np.linalg.solve(a,b) *0.1
print(x)