# -*- coding: utf-8 -*-
"""
Created on Sun May 10 00:36:19 2020

@author: SebastianYepes
"""
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
import pyodbc
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# X = loadmat('X.mat')
# Y = loadmat('Y.mat')
# x = X['X']
# y = Y['Y']

azuredriver = "ODBC Driver 17 for SQL Server"
azurebase = "sensores1"
usuario = "admin_sensores1"
password = "ITM_2020*"
server = "servidor-sensores1-2020--1.database.windows.net"
connStr = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+azurebase+';UID='+usuario+';PWD='+ password)
cursor = connStr.cursor()

# for index, row in enumerate(x):
#     insertar= '''INSERT INTO matriz2 (X1,X2,X3,X4,X5,X6,X7,X8) VALUES (?,?,?,?,?,?,?,?);'''
#     values = (row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7])
#     cursor.execute(insertar, values)
# connStr.commit()

# for index, row in enumerate(y):
#     insertar= '''INSERT INTO clase (clase) VALUES (?);'''
#     values = (int(row[0]))
#     cursor.execute(insertar, values)
# connStr.rcommit()

query = "SELECT * FROM dbo.matriz2 "
query2 ="SELECT * FROM dbo.clase"
X = pd.read_sql (query, connStr)
Y = pd.read_sql (query2, connStr)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=42)
Model = GaussianNB()
Model.fit(X_train,Y_train)
Y_es = Model.predict(X_test)
Y_es = np.expand_dims(Y_es,axis=1)
AccActual = accuracy_score(Y_test,Y_es)
print(classification_report(Y_test,Y_es))