# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 18:15:32 2023

@author: saga
"""

#******************************PYSPARK********************************
#https://www.youtube.com/watch?v=u6I8HCJlIk0&list=PLZoTAELRMXVNjiiawhzZ0afHcPvC8jpcg&index=5




#****************************CREAR SESION DE SPARK************************
import pyspark
import pandas as pd
from pyspark.sql import SparkSession # libreria necesaria para abrir sesion
from pyspark import SparkConf

# Read data from CSV file using pandas
pima = pd.read_csv(r'D:\RVC\Academico\PYTHON\Ejercicios\DTREE\diabetesXT.txt')

# Show headers
print(pima.head())
print()

#*HAY QUE ABIRAR UNA SESION DE SPARK
spark = SparkSession.builder.appName('EjemploA').getOrCreate()


#**************************CARGA DATOS EN CLUSTER************************
df_pyspark = spark.read.option('header','true').csv(r'D:\RVC\Academico\PYTHON\Ejercicios\DTREE\diabetesXT.txt')

#RESULTADO DATAFRAME CARGADO A SPARK
print(df_pyspark)
print()

#MUESTRA DE LAS CABECERAS
print(df_pyspark.head())
print()

#MUESTRA DE LA DATA CARGADA EN SPARK
print('DATA CARGADA EN SPARK')
df_pyspark.show()
print()

# DICCIONARIO DE DATOS
print('Diccionario de Datos')
df_pyspark.printSchema()
#PODEMOS VER QUE LA CLASE CAMBIA EN LA TABLA PYSPARK
print(type(pima))
print(type(df_pyspark))
print()
#*************HACER CONSULTAS A TABLA/FILTROS***************************

#FILTRO DE COLUMNA 
print('Filtro de columna, identificando a los pacientes con diabetes')
df_pyspark.filter("outcome= 1").show()
print()
#FILTRO DE COLUMNA CON SELECT
print('Filtro de columna con select de columnas')
df_pyspark.filter("outcome= 1").select(['Age','Glucose']).show()
print()

#FILTRO DE COLUMNA CON DOS CONDICIONES
print('Filtro de columna con DOS condiciones')
#METODO 01
#from pyspark.sql.functions import col
#df_pyspark.filter((col("Outcome") == 1) & (col("Age") >= 50)).show()

#METODO 02 CON | QUE SIGNIFICA "OR"
df_pyspark.filter((df_pyspark.Outcome == 1)|(df_pyspark.Age >= 50)).show()
print()

#METODO 03 CON ~ QUE SIGNIFICA "NOT"
print('Filtro de columna UTILIZANDO NOT "~" = 1')
df_pyspark.filter(~(df_pyspark.Outcome == 1)).show()
print()

#*************GROUP BY AND AGGREGATE FUNCTIONS***************************


# CAMBIAR TIPO DE DATO DE COLUMNA
from pyspark.sql.types import IntegerType, FloatType
df_pyspark = df_pyspark.withColumn("Pregnancies", df_pyspark["Pregnancies"].cast(IntegerType()))
df_pyspark = df_pyspark.withColumn("Glucose", df_pyspark["Glucose"].cast(IntegerType()))
df_pyspark = df_pyspark.withColumn("BloodPressure", df_pyspark["BloodPressure"].cast(IntegerType()))
df_pyspark = df_pyspark.withColumn("SkinThickness", df_pyspark["SkinThickness"].cast(IntegerType()))
df_pyspark = df_pyspark.withColumn("Insulin", df_pyspark["Insulin"].cast(IntegerType()))
df_pyspark = df_pyspark.withColumn("BMI", df_pyspark["BMI"].cast(FloatType()))
df_pyspark = df_pyspark.withColumn("DiabetesPedigreeFunction", df_pyspark["DiabetesPedigreeFunction"].cast(FloatType()))
df_pyspark = df_pyspark.withColumn("Age", df_pyspark["Age"].cast(IntegerType()))
df_pyspark = df_pyspark.withColumn("Outcome", df_pyspark["Outcome"].cast(IntegerType()))

print()
# DICCIONARIO DE DATOS
print('Diccionario de Datos')
df_pyspark.printSchema()
print()
#MUESTRA DE LA DATA CARGADA EN SPARK
print('DATA CARGADA EN SPARK')
df_pyspark.show()
print()

#*****************************GROUP BY*******************************************
#GROUP BY 
print('Promedio de Edad, Glucosa y Embarazos por tipos de pacientes')
df_pyspark.groupBy('Outcome').mean('age','glucose','Pregnancies').show()
print()
#COUNT
print('Cantidad de Pacientes por tipo de estado')
df_pyspark.groupBy('Outcome').count().show()
print()
#AGGREGATE
print('Promedio de Edad')
df_pyspark.agg({'Age':'mean'}).show()
print()


#***************************SPARK-MLlib**************************************
dlr_pyspark = spark.read.option('header','true').csv(r'D:\RVC\Academico\PYTHON\Ejercicios\PYSPARK\LRegresSparkDatasetxt.txt')

#RESULTADO DATAFRAME CARGADO A SPARK-EJERCICIO LREGRESSION
print(dlr_pyspark)
print()

#MUESTRA DE LAS CABECERAS
print(dlr_pyspark.head())
print()

#MUESTRA DE LA DATA CARGADA EN SPARK
print('DATA CARGADA EN SPARK')
dlr_pyspark.show()
print()
# DICCIONARIO DE DATOS
print('Diccionario de Datos')
dlr_pyspark.printSchema()
print()
# CAMBIAR TIPO DE DATO DE COLUMNA
from pyspark.sql.types import IntegerType, FloatType
dlr_pyspark = dlr_pyspark.withColumn("age", dlr_pyspark["age"].cast(IntegerType()))
dlr_pyspark = dlr_pyspark.withColumn("Experience", dlr_pyspark["Experience"].cast(IntegerType()))
dlr_pyspark = dlr_pyspark.withColumn("Salary", dlr_pyspark["Salary"].cast(IntegerType()))

# DICCIONARIO DE DATOS
print('Diccionario de Datos-Modificado')
dlr_pyspark.printSchema()

#*************************PASO-01: AGRUPAR VARIABLES INDEP*****************
from pyspark.ml.feature import VectorAssembler

featureassembler = VectorAssembler(inputCols=["age","Experience"],
                                   outputCol = "Independent_Features")

output = featureassembler.transform(dlr_pyspark)
output.show()

#*************************PASO-02: MATRIZ REGRESORA**********************

finalized_data = output.select("Independent_Features","Salary")
finalized_data.show()


#********************PASO-03: MODELO REGRESION LINEAL**********************

from pyspark.ml.regression import LinearRegression

train_data,test_data = finalized_data.randomSplit([0.60,0.40])
regressor = LinearRegression(featuresCol='Independent_Features',labelCol='Salary')
regressor =regressor.fit(train_data)

#Coeficientes
print('Coeficientes')
print(regressor.coefficients)
#Intercepto
print('INTERCEPTO')
print(regressor.intercept)

trainingSummary = regressor.summary
print()
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print()
print("r2: %f" % trainingSummary.r2)


#********************PASO-04: PREDICCION**********************

#PREDICTION-METODO 01
pred_results = regressor.evaluate(test_data)
pred_results.predictions.show()
print()
# Printing the R2 Score
print('R2-Score for train set:', trainingSummary.r2)
print('R2-Score for test set:', pred_results.r2)
print()

#PREDICTION-METODO 02
lr_predictions = regressor.transform(test_data)
lr_predictions.select("prediction","Salary","Independent_Features").show(5)
from pyspark.ml.evaluation import RegressionEvaluator
lr_evaluator = RegressionEvaluator(predictionCol="prediction", \
                 labelCol="Salary",metricName="r2")
print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(lr_predictions))

































