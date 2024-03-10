// Databricks notebook source
// Pregunta 4: ¿Qué puesto de Felicidad tiene el país con mayor GDP del 2020?

// Carga de dataframes y creacion de vistas temporales

val worldHappiness = spark.read.option("header", "true").option("inferSchema", "true").csv("dbfs:/FileStore/big-data-processing-practica/world_happiness_report.csv")

val worldHappiness2 = spark.read.option("header", "true").option("inferSchema", "true").csv("dbfs:/FileStore/big-data-processing-practica/world_happiness_report_2021.csv")

worldHappiness.createOrReplaceTempView("worldHappiness_temp")
worldHappiness2.createOrReplaceTempView("worldHappiness2_temp")

// COMMAND ----------

import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window

// Filtramos (filter) por año 2020 y creamos un ranking basado en "Life Ladder" con el metodo Window
val worldHappiness2020 = worldHappiness.filter(col("year") === "2020")
val windowLifeLadder = Window.orderBy(col("Life Ladder").desc)
val ranked2020 = worldHappiness2020.withColumn("Rank", rank().over(windowLifeLadder))

// Encontramos el pais con el mayor GDP ("Log GDP per capita") y su correspondiente puesto de felicidad
val countryHighestGDP = ranked2020.orderBy(col("Log GDP per capita").desc).first()
val countryName = countryHighestGDP.getAs[String]("Country name")
val rankByLifeLadder = countryHighestGDP.getAs[Int]("Rank")

println(s"El pais con el mayor GDP en 2020 es: $countryName y su puesto de felicidad es: $rankByLifeLadder")


// COMMAND ----------

// MAGIC %sql
// MAGIC SELECT 
// MAGIC     `Country name`, `Log GDP per capita`, `Life Ladder`,
// MAGIC     RANK() OVER (ORDER BY `Life Ladder` DESC) AS Position
// MAGIC FROM worldHappiness_temp
// MAGIC WHERE `year` = 2020
// MAGIC
// MAGIC
// MAGIC /* Otr forma de obtener el resultado con SQL*/
