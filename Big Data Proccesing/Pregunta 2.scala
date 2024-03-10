// Databricks notebook source
//PREGUNTA 2: ¿Cuál es el país más “feliz” del 2021 por continente (region) según la data?

// Carga de dataframes y creacion de vistas temporales

val worldHappiness = spark.read.option("header", "true").option("inferSchema", "true").csv("dbfs:/FileStore/big-data-processing-practica/world_happiness_report.csv")

val worldHappiness2 = spark.read.option("header", "true").option("inferSchema", "true").csv("dbfs:/FileStore/big-data-processing-practica/world_happiness_report_2021.csv")

worldHappiness.createOrReplaceTempView("worldHappiness_temp")
worldHappiness2.createOrReplaceTempView("worldHappiness2_temp")

// COMMAND ----------

// Definicion de un campo de agregacion con agrupacion, para obtener los valores unicos por region con su correspondiente valor maximo de "Ladder score"

import org.apache.spark.sql.functions.{col,max}

val maxLadderScoreByRegion = worldHappiness2.groupBy("Regional indicator").agg(max("Ladder score").alias("maxLadderScore"))

display(maxLadderScoreByRegion)

// COMMAND ----------

// Ejecutar el join de las tablas worldHappiness2 y maxLadderScoreByRegion a traves del campo "Regional Indicator", utilizando alias para evitar ambiguedades 

val result = worldHappiness2.as("wh2")
                    .join(maxLadderScoreByRegion.as("maxlsbr"), $"wh2.Regional indicator" === $"maxlsbr.Regional indicator")

//Luego aplicar metodo filter para obtener los paises que tienen el maximo Ladder Score por Region 

val countriesWithMaxLadderScoreByRegion = result.filter(col("wh2.Ladder score") === col("MaxLadderScore"))

display(countriesWithMaxLadderScoreByRegion.select("Country name", "maxlsbr.Regional indicator", "maxLadderScore"))
