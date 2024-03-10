// Databricks notebook source
//PREGUNTA 3: ¿Cuál es el país que más veces ocupó el primer lugar en todos los años?

// Carga de dataframes y creacion de vistas temporales

val worldHappiness = spark.read.option("header", "true").option("inferSchema", "true").csv("dbfs:/FileStore/big-data-processing-practica/world_happiness_report.csv")

val worldHappiness2 = spark.read.option("header", "true").option("inferSchema", "true").csv("dbfs:/FileStore/big-data-processing-practica/world_happiness_report_2021.csv")

worldHappiness.createOrReplaceTempView("worldHappiness_temp")
worldHappiness2.createOrReplaceTempView("worldHappiness2_temp")

// COMMAND ----------

// Usando codigo SQL en Spark, creamos la union entre las tablas worldHappiness_temp & worldHappiness2_temp para obtener un reporte consolidado (results) por todos los años (2005-2021). Para el 2021 se considera el valor de la columna "Life Ladder" igual a "Ladder score"  

val consolidatedReport = spark.sql(
  """
  SELECT `Country name`, 
       `year`,
       `Life Ladder`
  FROM worldHappiness_temp
  UNION
  SELECT `Country name`,
       2021 AS `year`,
       `Ladder score`
  FROM worldHappiness2_temp
  ORDER BY `Country name`, `Life Ladder` DESC
"""
)

display(consolidatedReport)

// COMMAND ----------

import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window

// Usando consolidatedReport, realizamos un ranking por pais (primer lugar) y conteo por año
val counted = consolidatedReport
  .withColumn("ranking", rank().over(Window.partitionBy("year").orderBy(col("Life Ladder").desc)))
  .filter(col("ranking") === 1)
  .withColumn("country_count", count("*").over(Window.partitionBy("year")))
  .select("Country name", "year", "Life Ladder", "country_count")

// Agrupamos y obtenemos el total de veces que un pais ocupo el primer lugar
val results = counted
  .groupBy("Country name")
  .agg(sum("country_count").alias("total_country_count"))

results.orderBy($"total_country_count".desc).show()

// En periodo entre 2005 al 2021 se obtiene un empate en el primer lugar entre Finland y Denwark.  

