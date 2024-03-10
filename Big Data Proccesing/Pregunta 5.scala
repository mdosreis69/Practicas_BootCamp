// Databricks notebook source
//Pregunta 5: ¿En que porcentaje a variado a nivel mundial el GDP promedio del 2020 respecto al 2021? ¿Aumentó o disminuyó?

// Carga de dataframes y creacion de vistas temporales

val worldHappiness = spark.read.option("header", "true").option("inferSchema", "true").csv("dbfs:/FileStore/big-data-processing-practica/world_happiness_report.csv")

val worldHappiness2 = spark.read.option("header", "true").option("inferSchema", "true").csv("dbfs:/FileStore/big-data-processing-practica/world_happiness_report_2021.csv")

worldHappiness.createOrReplaceTempView("worldHappiness_temp")
worldHappiness2.createOrReplaceTempView("worldHappiness2_temp")

// COMMAND ----------

// Calculo del GDP promedio para 2020 y 2021

import org.apache.spark.sql.functions.{avg}

val avgGDP2020 = worldHappiness.agg(avg("Log GDP per capita")).first().getDouble(0)
val avgGDP2021 = worldHappiness2.agg(avg("Logged GDP per capita")).first().getDouble(0)

// Calculo de variacion entre 2020 y 2021
val variationYears = ((avgGDP2021 - avgGDP2020) / avgGDP2020) * 100
println(s"El porcentaje de variacion del GDP fue de: $variationYears%")

// Comparacion de los resultados
val comparisonResults = if (avgGDP2020 > avgGDP2021) {
  "El GDP promedio disminuyo del 2020 al 2021"
} else if (avgGDP2020 < avgGDP2021) {
  "El GDP promedio aumento del 2020 al 2021"
} else {
  "El GDP promedio se mantuvo igual entre 2020 y 2021."
}

println(comparisonResults)

