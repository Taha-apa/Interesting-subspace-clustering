# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 14:26:54 2022

@author: TAHA
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_cg
plt.style.use(['ggplot'])
def Regression(fileName,maxIteration):
    X,Y = get_data_from_file(fileName)
    theta = create_theta(X, Y)
    #create a plot of data
    plt.scatter(X[:,1],X[:,2],c=Y,marker  = 'o',linestyle = 'None')
    #using fmin_congucate to find min of cost function by cost function and it's derivitives
    res_1 = fmin_cg(calculateCost, x0 = theta,fprime=calculateDerivitiveOfCost,args=(X,Y),maxiter=maxIteration)
    plotDecisionBoundary(X.astype(int), res_1)
    
def get_data_from_file(file):
    X=[]
    Y=[]
    X = np.loadtxt(file,delimiter=',',usecols=(0,1))
    X = np.insert(X,0,[1 for i in range(len(X))], axis=1)
    Y = np.loadtxt(file,delimiter=',',usecols=2).reshape((X.shape[0],1))
    return X,Y
def getDataRange(X):
    """
    output min-max X Collumn 1 - Column 2
    """
    xMin = min(X[:,1]), min(X[:,2])
    xMax = max(X[:,1]) , max(X[:,2])
    return xMin,xMax
def create_theta(X,Y):
    return np.zeros((1,X.shape[1]) if all(Y.shape) else np.zeros((Y.shape[1],X.shape[1])))

def sigmoid(x):
    return 1/(1+np.exp(-x))

def mapFeature(X,degree):
    """
    ASSUMPTION : X HAS 2 Dimentions
    """
    Xmapped = X
    for i in range(degree):
        for j in range(i):
            np.append(Xmapped,(X[1]^i).dot(X[2]^(j-i)))
    return Xmapped

def calculateHypothesis(theta,X):
    return sigmoid(np.sum(theta * X,axis=1,keepdims=True))

def calculateCost(theta,X,Y):
    Hypothesis = calculateHypothesis(theta, X)
    amountOfRecords = np.size(Hypothesis)
    return cost(amountOfRecords, Hypothesis, Y)

def cost(amountOfRecords,Hypothesis,Y):
    return (-1/amountOfRecords) * np.sum(Y*np.log(Hypothesis)+(1-Y)*np.log(1-Hypothesis))

def calculateDerivitiveOfCost(theta,X,Y):
    Hypothesis = calculateHypothesis(theta, X)
    return derivitiveOfCost(Hypothesis, X, Y)

def derivitiveOfCost(Hypothesis,X,Y):
    amountOfRecords = np.size(Hypothesis)
    return ((1/amountOfRecords) * (Hypothesis-Y).T.dot(X)).reshape(X.shape[1],)

def checkTestData(X,theta):
    return sigmoid(np.sum(theta.reshape(1,theta.shape[0]) * X,axis=1,keepdims=True))

def plotDecisionBoundary(xAsInt,optTheta):
    """
    AccuracyRate : must be greater than 1 and it must be an integer
    due to the dimensions of theta in this example, the classifier creates a linear deciesion boundary
    so we only need 2 points with 0.5 output to draw the deciesion boundary
    """
    xRangeInt  = getDataRange(xAsInt)
    points = getBoundaryPoints(xRangeInt, optTheta)    
    plt.plot([points[0][0],points[len(points)-1][0]],[points[0][1],points[len(points)-1][1]])
    
def getBoundaryPoints(xRangeInt,optTheta):
    points=[]    
    for i in range(xRangeInt[0][0],xRangeInt[1][0]):
        for j in range(xRangeInt[0][1],xRangeInt[1][1]):
            if(isBoundaryPoint(optTheta, [1,i,j])):
                points.append([i,j])
    return points
                
def isBoundaryPoint(optTheta,point):
    checkTestDataAtThePoint = float(checkTestData(point, optTheta))
    return 0.4 <= checkTestDataAtThePoint  <= 0.6
        
Regression("ex2data1.txt",1000)
"""
the code commented down below can be used to test a point given by user:
"""
#inputPoint =[1] + [int(i) for i in input("Enter Point: ").split(',')]
#print(checkTestData(inputPoint, res_1))
