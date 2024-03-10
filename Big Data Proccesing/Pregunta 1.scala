// Databricks notebook source
// PREGUNTA 1: ¿Cuál es el país más “feliz” del 2021 según la data? (considerar la columna “Ladder score”, mayor número más feliz es el país)

// Carga de datos

val worldHappiness = spark.read.option("header", "true").option("inferSchema", "true").csv("dbfs:/FileStore/big-data-processing-practica/world_happiness_report.csv")

val worldHappiness2 = spark.read.option("header", "true").option("inferSchema", "true").csv("dbfs:/FileStore/big-data-processing-practica/world_happiness_report_2021.csv")


// COMMAND ----------

worldHappiness.printSchema

// COMMAND ----------

worldHappiness2.printSchema

// COMMAND ----------

// Crear listas temporales para ejecutar funciones SQL en Spark

worldHappiness.createOrReplaceTempView("worldHappiness_temp")
worldHappiness2.createOrReplaceTempView("worldHappiness2_temp")

spark.catalog.listTables.show

// COMMAND ----------

// MAGIC %sql
// MAGIC SELECT * 
// MAGIC FROM worldHappiness2_temp
// MAGIC
// MAGIC /*Explorando dataframe reporte 2021*/

// COMMAND ----------

// MAGIC %sql
// MAGIC SELECT `Country name`, `Ladder score` 
// MAGIC FROM worldHappiness2_temp
// MAGIC ORDER BY "Ladder Score" DESC
// MAGIC LIMIT 1;
// MAGIC
// MAGIC /*Query para saber el país más “feliz” del 2021 segun el “Ladder score” */
// MAGIC

// COMMAND ----------

// Otra forma de obtener la respuesta con codigo scala y la funcion max

import org.apache.spark.sql.functions.{col,max}

val maxLadderScore = worldHappiness2.select(max(col("Ladder score"))).first()(0).asInstanceOf[Double]

display(worldHappiness2.where(col("Ladder score") === maxLadderScore))

