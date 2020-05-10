# -*- coding: utf-8 -*-
"""
Created on Fri May  8 11:39:15 2020

@author: SebastianYepes
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import pyodbc  
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import json
import logging
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, accuracy_score

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

connStr.close()

X = X.to_numpy()
Y = Y.to_numpy()

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=42)
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, Y_train)
prediccion = neigh.predict(X_test)
AccActual = accuracy_score(Y_test,prediccion)

json_response = json.dumps(classification_report(Y_test, prediccion),indent=2)

 