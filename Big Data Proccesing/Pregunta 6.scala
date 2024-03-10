// Databricks notebook source
//Pregunta 6: ¿Cuál es el país con mayor expectativa de vida (“Healthy life expectancy at birth”)? Y ¿Cuánto tenia en ese indicador en el 2019?

// Carga de dataframes y creacion de vistas temporales

val worldHappiness = spark.read.option("header", "true").option("inferSchema", "true").csv("dbfs:/FileStore/big-data-processing-practica/world_happiness_report.csv")

val worldHappiness2 = spark.read.option("header", "true").option("inferSchema", "true").csv("dbfs:/FileStore/big-data-processing-practica/world_happiness_report_2021.csv")

worldHappiness.createOrReplaceTempView("worldHappiness_temp")
worldHappiness2.createOrReplaceTempView("worldHappiness2_temp")

// COMMAND ----------

//Calcular el maximo para el campo "Healthy life expectancy at birth" y obtener el nombre del pais

import org.apache.spark.sql.functions._

val maxHealthyLife = worldHappiness.agg(max("Healthy life expectancy at birth")).collect()(0)(0)

val countryMaxHealthyLife = worldHappiness.filter(col("Healthy life expectancy at birth") === maxHealthyLife).select("Country name")

countryMaxHealthyLife.show()

// COMMAND ----------

//Calculo indicador para el año 2019

import org.apache.spark.sql.functions.{col, desc}

val countryHealthyLife = worldHappiness
  .select("Country name", "year", "Healthy life expectancy at birth")
  .where(col("year") === "2019")
  .orderBy(desc("Healthy life expectancy at birth"))
  .limit(1)

countryHealthyLife.show()
