// Databricks notebook source
// Carga de datos

val worldHappiness = spark.read.option("header", "true").option("inferSchema", "true").csv("dbfs:/FileStore/big-data-processing-practica/world_happiness_report.csv")

val worldHappiness2 = spark.read.option("header", "true").option("inferSchema", "true").csv("dbfs:/FileStore/big-data-processing-practica/world_happiness_report_2021.csv")

// COMMAND ----------

// Crear listas temporales para ejecutar funciones SQL en Spark

worldHappiness.createOrReplaceTempView("worldHappiness_temp")
worldHappiness2.createOrReplaceTempView("worldHappiness2_temp")

spark.catalog.listTables.show

// COMMAND ----------

//Explorando dataframe 
worldHappiness2.printSchema

// COMMAND ----------

// PREGUNTA 1: ¿Cuál es el país más “feliz” del 2021 según la data? (considerar la columna “Ladder score”, mayor número más feliz es el país)

// COMMAND ----------

// MAGIC %sql
// MAGIC SELECT `Country name`, `Ladder score` 
// MAGIC FROM worldHappiness2_temp
// MAGIC ORDER BY "Ladder Score" DESC
// MAGIC LIMIT 1;
// MAGIC
// MAGIC /*Query para saber el país más “feliz” del 2021 segun el “Ladder score” */

// COMMAND ----------

// Otra forma de obtener la respuesta con codigo scala y la funcion max

import org.apache.spark.sql.functions.{col,max}

val maxLadderScore = worldHappiness2.select(max(col("Ladder score"))).first()(0).asInstanceOf[Double]

display(worldHappiness2.where(col("Ladder score") === maxLadderScore))

// COMMAND ----------

//PREGUNTA 2: ¿Cuál es el país más “feliz” del 2021 por continente (region) según la data?

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

// COMMAND ----------

//PREGUNTA 3: ¿Cuál es el país que más veces ocupó el primer lugar en todos los años?

// Usando codigo SQL en Spark, creamos la union entre las tablas worldHappiness_temp & worldHappiness2_temp para obtener un reporte consolidado (results) por todos los años (2005-2021). Para el 2021 se considera el valor de la columna "Life Ladder" igual a valor de "Ladder score"  

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

// COMMAND ----------

// Pregunta 4: ¿Qué puesto de Felicidad tiene el país con mayor GDP del 2020?

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
// MAGIC /* Otra forma de obtener el resultado con SQL*/

// COMMAND ----------

// Calculo del GDP promedio para 2020 y 2021

import org.apache.spark.sql.functions.{avg}

//val avgGDP2020 = worldHappiness.agg(avg("Log GDP per capita")).first().getDouble(0)
val filterWH2020 = worldHappiness.filter("year = 2020")
val avgGDP2020 = filterWH2020.agg(avg("Log GDP per capita")).first().getDouble(0)
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

// COMMAND ----------

//Pregunta 6: ¿Cuál es el país con mayor expectativa de vida (“Healthy life expectancy at birth”)? Y ¿Cuánto tenia en ese indicador en el 2019?

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
