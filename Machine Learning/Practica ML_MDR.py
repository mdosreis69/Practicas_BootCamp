#!/usr/bin/env python
# coding: utf-8

# # Práctica de ML con Python
# ### Predecir el precio de alquiler de Airbnb
# 
# El objetivo de la práctica es estudiar un problema de Machine Learning realista siguiendo la metodología y buenas prácticas explicadas durante las clases. Haremos primero un análisis exploratorio para familiarizarnos con la base de datos de Airbnb
# 
# Luego aplicaremos los siguientes pasos para predecir el precio de alquiler en la ciudad de Madrid:
# 
#     - Aplicar técnicas de procesamiento/transformación de variables 
#     - Identificar los modelos
#     - Las variables potencialmente más relevantes y 
#     - La métrica adecuada para contrastar los distintos modelos. 
# 

# In[223]:


# Lo primero es cargar las librerías y funciones necesarias

import numpy  as np  
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# ## 1. Carga de datos y filtrado inicial por ciudad

# In[224]:


# Cargamos el fichero
   
airbnb_data = pd.read_csv('./data/airbnb-listings-extract.csv',sep=';', decimal='.')
print(airbnb_data.shape)
airbnb_data.head(5)


# El estudio se limitara a Madrid, por cual vamos a realizar un filtrado por 'Country' = "Spain" y 'Market' = "Madrid" 
# para abarcar la mayor cantidad de registros de la ciudad

# In[225]:


airbnb_data = airbnb_data[(airbnb_data['Country'] == "Spain") & (airbnb_data['Market'] == "Madrid")]
airbnb_data.shape


# Al considerar solo registros de Madrid, podemos eliminar las columnas que tienen que ver con la identificación 
# de la ciudad, estado, país, etc. (City/State/Market/Smart Location/Country Code/Country) antes de dividir en train y test

# In[226]:


airbnb_data = airbnb_data.drop(columns= [
'City', 'State', 'Market', 'Smart Location', 'Country Code', 'Country'])
airbnb_data.shape


# ## 2. Preparación de datos 
# Realizamos la división de los datos utilizando el método train_test_split de sklearn

# In[227]:


# División train/test (80%/20%)
from sklearn.model_selection import train_test_split

train, test = train_test_split(airbnb_data, test_size=0.2, shuffle=True, random_state=0)

print(f'Dimensiones del dataset de training: {train.shape}')
print(f'Dimensiones del dataset de test: {test.shape}')

# Guardamos
train.to_csv('./data/airbnb_train.csv', sep=';', decimal='.', index=False)
test.to_csv('./data/airbnb_test.csv', sep=';', decimal='.', index=False)

# A partir de este momento cargamos y trabajamos solo con el dataset de train 

airbnb_train = pd.read_csv('./data/airbnb_train.csv', sep=';', decimal='.')


# ## 3. Análisis exploratorio
# 
# Revisamos el dataset de train utilizando las funciones exploratorias de Pandas

# In[228]:


airbnb_train.head(5)


# In[229]:


airbnb_train.describe().T


# In[230]:


airbnb_train.info()


# ### Pre-procesamiento de variables
# Realicemos los siguientes pasos de tratamiento y transformación de variables:
# - Eliminación de variables que no son relevantes para el estudio
# - Imputación de valores ausentes
# - Codificación de variables categóricas
# - Transformación de variables
# - Escalado/Normalización

# #### - Eliminación de variables 

# In[231]:


# Eliminamos las columnas con IDs, Names y URLs ya que no aportan mayor información para el análisis del precio

airbnb_train = airbnb_train.drop(columns= [
    'ID', 'Scrape ID', 'Host ID','Host Name', 'Name','Host Location',
    'Listing Url', 'Thumbnail Url','Medium Url', 'Picture Url',
    'XL Picture Url', 'Host URL', 'Host Thumbnail Url','Host Picture Url'
])
airbnb_train.shape


# In[232]:


# Analicemos las columnas que tienen todos sus valores faltantes o inexistentes 

for column in airbnb_train.columns:
    unique_values = airbnb_train[column].unique()
    if len(unique_values) == 1:
        print(f"Column '{column}' tiene este unico valor: {unique_values[0]}")


# In[233]:


# Eliminamos esas columnas

airbnb_train = airbnb_train.drop(columns= [
    'Experiences Offered', 'Host Acceptance Rate', 'Has Availability','Jurisdiction Names'])
airbnb_train.shape


# También podemos eliminar las columnas que contienen texto, notas, instrucciones y evaluaciones sobre el inmueble y huesped. 
# Porque no pueden ser clasificadas en tipo binario o en clases que tengan algún sentido para el estudio del precio

# In[234]:


airbnb_train = airbnb_train.drop(columns= [
    'Summary', 'Space', 'Description', 'Neighborhood Overview',  'Notes', 'Transit', 'Access', 'Interaction', 
    'House Rules', 'Host About', 'Host Neighbourhood', 'Host Verifications', 'Street', 'Amenities', 'Features'
])
airbnb_train.shape


# Algunas variables contienen o engloban la misma informacion ( son repetitivas), podemos quedarnos con solo una de ellas.
# Por ejemplo 'Host Response Rate'/'Host Response Time'(eliminado), 'Host Total Listings Count'/'Host Listings Count'(e) 
# 'Calculated host listings count'(e), 'Latitude'-'Longitude'/'Geolocation'(e). También 'Zipcode' no aporta mayor
# información, la latitud/longitud tiene mayor precision en la ubicación del inmueble.

# In[235]:


airbnb_train = airbnb_train.drop(columns= ['Host Response Time', 'Host Listings Count', 'Calculated host listings count', 
                                           'Geolocation', 'Zipcode'
                                          ])
airbnb_train.shape


# Analicemos las columnas que hacen referencias a fechas (registro, actualizaciones y encuestas) para saber si son
# relevantes para el estudio del precio. Por ejemplo, alguna que nos permita saber o calcular la antiguedad del inmueble.

# In[236]:


sorted_df = airbnb_train.sort_values(by='Last Scraped')
sorted_df['Last Scraped'].head(10)


# In[237]:


sorted_df = airbnb_train.sort_values(by='Last Scraped', ascending=False)
sorted_df['Last Scraped'].head(10)

# Solo valores entre marzo y abril de 2017, parece refererirse a fechas de extracción de información mediante técnicas scraping


# 'Host since' abarca mayor información (2009-2017) e indica desde cuando el anfitrión esta en la plataforma de Airbnb, 
# pero no ayuda a saber la antiguedad de inmueble. 

# In[238]:


sorted_df = airbnb_train.sort_values(by='Host Since')
sorted_df['Host Since']


# In[239]:


sorted_df = airbnb_train.sort_values(by='Calendar Updated')
sorted_df['Calendar Updated']

# Hace referencia a actualizaciones en el calendario, no es relevante 


# In[240]:


sorted_df = airbnb_train.sort_values(by='First Review')
sorted_df['First Review']


# In[241]:


sorted_df = airbnb_train.sort_values(by='Last Review')
sorted_df['Last Review']

# Fechas asociadas a encuestas de satisfacción: 'First Review'(2010-2017) y 'Last Review' (2012-2017) no son importantes.


# In[242]:


# Según los criterios mencionados, eliminamos las columnas de fechas

airbnb_train = airbnb_train.drop(columns= [
    'Host Since', 'Last Scraped', 'Calendar Updated', 'Calendar last Scraped', 'First Review', 'Last Review'
])
airbnb_train.shape


# La variable 'License' tiene muchos registros en blanco, pocas filas con un valor (195), que a veces empieza con las letras
# "VT-" pero luego no sigue un patrón y varia mucho en caracteres. Podemos eliminar esa columna

# In[243]:


airbnb_train = airbnb_train.drop(columns= ['License'])
airbnb_train.shape


# La variable 'Square Feet' tiene muchos registros sin valor (10202), un 96.09%. Podemos eliminarla no ayuda para el análisis

# In[244]:


airbnb_train = airbnb_train.drop(columns= ['Square Feet'])
airbnb_train.shape


# #### Imputación de valores 

# La columna 'Neighbourhood' abarca la información sobre el barrio, pero tiene varios registros en blanco. 
# Podemos usar el valor de la columna adjacente 'Neighbourhood Cleansed' que mantiene relación directa para llenarlos 
# en caso de ausencia.

# In[245]:


print(airbnb_train.loc[:10, 'Neighbourhood'])
print(airbnb_train['Neighbourhood'].isnull().sum())


# In[246]:


airbnb_train['Neighbourhood'] = airbnb_train['Neighbourhood'].fillna(airbnb_train['Neighbourhood Cleansed'])
print(airbnb_train.loc[:10, 'Neighbourhood'])


# In[247]:


# Eliminamos 'Neighbourhood Cleansed', 'Neighbourhood Group Cleansed', la columna 'Neighbourhood' ya abarca la info mas completa del barrio.  
airbnb_train = airbnb_train.drop(columns= ['Neighbourhood Cleansed', 'Neighbourhood Group Cleansed'])
airbnb_train.shape


# Similar caso con La columna 'Review Scores Rating', que recopila la puntuación general de satisfacción del cliente (escala del 0-100) pero tiene varios registros vacios. En este caso, podemos completar los valores faltantes con la media.

# In[248]:


print(airbnb_train['Review Scores Rating'].isnull().sum())
print("La media de 'Review Scores Rating' es:", airbnb_train['Review Scores Rating'].mean())


# In[249]:


airbnb_train['Review Scores Rating'].fillna(airbnb_train['Review Scores Rating'].mean(), inplace=True)


# Adicionalmente la columna 'Review Scores Rating' engloba de cierta manera los otros reviews sobre 
# limpieza, localizacion. etc., por lo tanto podemos simplificar y eliminar las otras columnas con esta subcategorias

# In[250]:


airbnb_train = airbnb_train.drop(columns= [
    'Review Scores Accuracy', 'Review Scores Cleanliness', 'Review Scores Checkin',
    'Review Scores Communication', 'Review Scores Location', 'Review Scores Value'
])
airbnb_train.shape


# Vamos avanzar con el análisis antes de imputar mas variables, puede que se eliminen algunas por los próximos pasos

# #### - Codificación de variables categóricas
# Podemos observar que todavia quedan variables no númericas en el dataset de Train, así que podemos codificarlas utilizando el metodo del LabelEncoder de sklearn antes avanzar con las siguientes etapas del analisis. 

# In[251]:


# Primer caso el barrio 'Neighbourhood':

from sklearn.preprocessing import LabelEncoder

le_Neighbourhood = LabelEncoder()

airbnb_train['Neighbourhood'] = le_Neighbourhood.fit_transform(airbnb_train['Neighbourhood'])
le_Neighbourhood.classes_


# In[252]:


# Siguientes variables categóricas

from sklearn.preprocessing import LabelEncoder

le_property = LabelEncoder()
le_Room = LabelEncoder()
le_bed = LabelEncoder()
le_Cancellation = LabelEncoder()

airbnb_train['Property Type'] = le_property.fit_transform(airbnb_train['Property Type'])
airbnb_train['Room Type'] = le_Room.fit_transform(airbnb_train['Room Type'])
airbnb_train['Bed Type'] = le_bed.fit_transform(airbnb_train['Bed Type'])
airbnb_train['Cancellation Policy'] = le_Cancellation.fit_transform(airbnb_train['Cancellation Policy'])

print(le_property.classes_)
print(le_Room.classes_)
print(le_bed.classes_)
print(le_Cancellation.classes_)


# In[253]:


airbnb_train.info()


# #### Transformacion de variables 

# ## 4. Visualizaciones y más análisis
# Analisis de las variables por separado: Con histogramas podemos conocer como son las distribuciones de cada una de las variables y adicionalmente ver si tienen datos anómalos (outliers) que debamos tratar.
# 
# Relaciones entre variables: Con scatter plot y correlación podemos entender las relaciones entre cada una de las variables independientes y con la variable de interés (precio).

# In[254]:


plt.figure(figsize=(25,35))

for i,feature in enumerate(airbnb_train.columns):
    plt.subplot(10,3,i+1)
    plt.hist(airbnb_train.loc[:,feature],density=0, alpha=0.75, label='frecuency')
    plt.legend()
    plt.title(feature)

plt.show()


# Suena lógico que el precio semanal y precio mensual este muy correlacionado con la variable objetivo de precio, veamos esto con un scatter plot para confirmar que podemos eliminar estas columnas

# In[255]:


airbnb_train.plot(kind = 'scatter',x='Weekly Price',y = 'Price')
plt.xlabel('# Weekly Price')
plt.ylabel('Price ($)')
plt.show()


# In[256]:


airbnb_train.plot(kind = 'scatter',x='Monthly Price',y = 'Price')
plt.xlabel('# Monthly Price')
plt.ylabel('Price ($)')
plt.show()


# In[257]:


# Eliminamos Weekly Price y Monthly Price

airbnb_train = airbnb_train.drop(columns= ['Weekly Price', 'Monthly Price'])
airbnb_train.shape


# Generemos la matriz de correlación entre variables

# In[258]:


# Graficamos la matrix de correlación para observar el nivel de correlación entre variables

import seaborn as sns

corr = np.abs(airbnb_train.drop(['Price'], axis=1).corr())

mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(20, 10))

sns.heatmap(corr, mask=mask,vmin = 0.0, vmax=1.0, center=0.5,
            linewidths=.1, cmap="YlGnBu", cbar_kws={"shrink": .8})

plt.show()


# Podemos ver algunas variables altamente correlacionadas como:
# 
# - 'Beds' vs 'Accommodates'
# - 'Availability 60' vs 'Availability 30'
# - 'Availability 90' vs 'Availability 60'
# - 'Availability 90' vs 'Availability 30'
# 
# Lo cual puede producir un problema de colinealidad, afectando los algoritmos de ML que usemos en el análisis.
# 
# Tenemos otras pero en menor grado: 
# - 'Beds' vs 'Bedrooms'
# - 'Bedrooms' vs 'Accommodates'
# - 'Reviews per Month' vs 'Number of Reviews'
# 
# Evaluemos el coeficiente de correlación (𝜌) entre estos atributos fijando un umbral superior (0.9) como criterio 
# para saber si descartartamos alguna columna adicional. 

# In[259]:


corr_matrix = airbnb_train.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

threshold = 0.9
pairs = np.where(upper>threshold)
fx = airbnb_train.columns[pairs[0]]
fy = airbnb_train.columns[pairs[1]]

i=1
plt.figure(figsize=(22,5))
for f1,f2 in zip(fx,fy):
    
    plt.subplot(1,5,i)
    
    plt.scatter(airbnb_train[f1],airbnb_train[f2], c=airbnb_train['Price'], alpha=0.25)
    plt.xlabel(f1)
    plt.ylabel(f2)
    plt.grid()
    plt.tight_layout()
    
    i+=1
    
plt.show()


# In[260]:


# Para evitar estas correlaciones altas en disponibilidad, eliminamos

airbnb_train = airbnb_train.drop(columns= ['Availability 60','Availability 90'])
airbnb_train.shape


# In[261]:


# Veamos el resto de columnas con registros sin valores (NaN)
nan_counts = airbnb_train.isna().sum()

for column, count in nan_counts.items():
    print(f'{column}: {count}')


# In[262]:


# Imputación de valores faltantes utilizando la media para las variables que siguen una distribución normal o parecida

airbnb_train['Host Response Rate'].fillna(airbnb_train['Host Response Rate'].median(), inplace=True)
airbnb_train['Host Total Listings Count'].fillna(airbnb_train['Host Total Listings Count'].median(), inplace=True)
airbnb_train['Price'].fillna(airbnb_train['Price'].median(), inplace=True)
airbnb_train['Security Deposit'].fillna(airbnb_train['Security Deposit'].median(), inplace=True)
airbnb_train['Cleaning Fee'].fillna(airbnb_train['Cleaning Fee'].median(), inplace=True)
airbnb_train['Reviews per Month'].fillna(airbnb_train['Reviews per Month'].median(), inplace=True)

# Imputación de valores faltantes utilizando la moda para las variables que son categoricas o siguen valores discretos
airbnb_train['Bathrooms'].fillna(airbnb_train['Bathrooms'].mode()[0], inplace=True)
airbnb_train['Bedrooms'].fillna(airbnb_train['Bedrooms'].mode()[0], inplace=True)
airbnb_train['Beds'].fillna(airbnb_train['Beds'].mode()[0], inplace=True)


# In[263]:


print(airbnb_train.isna().sum())


# #### Eliminación de outliers
# 
# Viendo los histogramas de arriba podemos ver posibles outliers en algunas variables, como Bedrooms, Accommodates, Beds, Bathrooms y Security Deposit. Usemos  scatter plot y `value_counts`para estudiarlos.

# In[264]:


#Caso Bedrooms

airbnb_train['Bedrooms'].value_counts().sort_index(ascending=True)


# In[265]:


airbnb_train.plot(kind = 'scatter',x='Bedrooms',y = 'Price')
plt.xlabel('# Bedrooms')
plt.ylabel('Price ($)')
plt.show()


# Evaluemos la perdida de datos si eliminamos los outliers de Bedrooms por encima de 8

# In[266]:


no_outliers_bedrooms = airbnb_train[airbnb_train['Bedrooms'] <=8]

print(
    f'Original: {airbnb_train.shape[0]} // '
    f'Modificado: {no_outliers_bedrooms.shape[0]}\nDiferencia: {airbnb_train.shape[0] - no_outliers_bedrooms.shape[0]}'
)
print(f'Variación: {(((airbnb_train.shape[0] - no_outliers_bedrooms.shape[0])/airbnb_train.shape[0])*100):.2f}%')


# In[267]:


# Caso Accommodates

airbnb_train['Accommodates'].value_counts().sort_index(ascending=True)


# In[268]:


#Caso Beds

airbnb_train['Beds'].value_counts().sort_index(ascending=True)


# En ambos casos no parece haber outliers bien definidos

# In[269]:


#Caso Bathrooms

airbnb_train['Bathrooms'].value_counts().sort_index(ascending=True)


# In[270]:


airbnb_train.plot(kind = 'scatter',x='Bathrooms',y = 'Price')
plt.xlabel('# Bathrooms')
plt.ylabel('Price ($)')
plt.show()


# In[271]:


no_outliers_bathrooms = airbnb_train[airbnb_train['Bathrooms'] <=7]

print(
    f'Original: {airbnb_train.shape[0]} // '
    f'Modificado: {no_outliers_bathrooms.shape[0]}\nDiferencia: {airbnb_train.shape[0] - no_outliers_bathrooms.shape[0]}'
)
print(f'Variación: {(((airbnb_train.shape[0] - no_outliers_bathrooms.shape[0])/airbnb_train.shape[0])*100):.2f}%')


# In[272]:


# Caso Security Deposit

airbnb_train['Security Deposit'].value_counts().sort_index(ascending=True)


# In[273]:


airbnb_train.plot(kind = 'scatter',x='Security Deposit',y = 'Price')
plt.xlabel('Security Deposit')
plt.ylabel('Price ($)')
plt.show()


# In[274]:


no_outliers_security = airbnb_train[airbnb_train['Security Deposit'] <=900.0]

print(
    f'Original: {airbnb_train.shape[0]} // '
    f'Modificado: {no_outliers_security.shape[0]}\nDiferencia: {airbnb_train.shape[0] - no_outliers_security.shape[0]}'
)
print(f'Variación: {(((airbnb_train.shape[0] - no_outliers_security.shape[0])/airbnb_train.shape[0])*100):.2f}%')


# Eliminamos son los outliers encontrados de 'Security Deposit' y 'Bathrooms' que representan un valor muy bajo 0.02%. El resto de las columnas no las tocamos, porque la eliminacion podría hacer perder muestras significativas si alguna variable resulta ser predictora del precio.

# In[275]:


airbnb_train = airbnb_train[airbnb_train['Security Deposit'] <=900.0]
airbnb_train = airbnb_train[airbnb_train['Bathrooms'] <=7]


# In[276]:


airbnb_train.shape


# ## 4. Generación de nuevas características
# 
# Nuevas variables que pueden tener sentido para entender mejor el entorno del inmueble y que sean relevantes para calcular el precio del inmueble: 
# 
# - Relación entre Bedrooms y Bathrooms.
# - Elevar al cuadrado el número de Bedrooms
# - Relación entre Accomodates y Beds

# In[277]:


# Relación entre Bedrooms y Bathrooms, apliquemos un producto

# airbnb_train['capacity'] = airbnb_train['Bathrooms'] + airbnb_train['Bedrooms']
airbnb_train['bathrooms_per_bedrooms'] = airbnb_train['Bathrooms'] * airbnb_train['Bedrooms']


# In[278]:


# Elevar al cuadrado el número de Bedrooms

airbnb_train['bedrooms_squared'] = airbnb_train['Bedrooms'].apply(lambda x: x**2)


# In[279]:


# Relación entre Accomodates y Beds

airbnb_train['beds_per_accommodates'] = airbnb_train['Beds'] / airbnb_train['Accommodates']


# In[280]:


airbnb_train.info()


# ## 5. Modelado, cross-validation y estudio de resultados en train y test
# 
# Cargamos nuevamente los datos de train / test y aplicamos las mismas transformaciones.

# In[281]:


# Carga de datos de Train
    
house_data_train = pd.read_csv('./data/airbnb_train.csv',sep=';', decimal='.')

# Imputación

house_data_train['Neighbourhood'] = house_data_train['Neighbourhood'].fillna(house_data_train['Neighbourhood Cleansed'])
house_data_train['Review Scores Rating'].fillna(house_data_train['Review Scores Rating'].mean(), inplace=True)
house_data_train['Host Response Rate'].fillna(house_data_train['Host Response Rate'].median(), inplace=True)
house_data_train['Host Total Listings Count'].fillna(house_data_train['Host Total Listings Count'].median(), inplace=True)
house_data_train['Price'].fillna(house_data_train['Price'].median(), inplace=True)
house_data_train['Security Deposit'].fillna(house_data_train['Security Deposit'].median(), inplace=True)
house_data_train['Cleaning Fee'].fillna(house_data_train['Cleaning Fee'].median(), inplace=True)
house_data_train['Reviews per Month'].fillna(house_data_train['Reviews per Month'].median(), inplace=True)
house_data_train['Bathrooms'].fillna(house_data_train['Bathrooms'].mode()[0], inplace=True)
house_data_train['Bedrooms'].fillna(house_data_train['Bedrooms'].mode()[0], inplace=True)
house_data_train['Beds'].fillna(house_data_train['Beds'].mode()[0], inplace=True)

# Eliminamos las columnas no importantes segun analisis previo

house_data_train = house_data_train.drop(columns= [
    'ID', 'Scrape ID', 'Host ID','Host Name', 'Name','Host Location',
    'Listing Url', 'Thumbnail Url','Medium Url', 'Picture Url',
    'XL Picture Url', 'Host URL', 'Host Thumbnail Url','Host Picture Url',
    'Experiences Offered', 'Host Acceptance Rate', 'Has Availability',
    'Jurisdiction Names', 'Summary', 'Space', 'Description', 'Neighborhood Overview',
    'Notes', 'Transit', 'Access', 'Interaction', 'House Rules', 'Host About', 
    'Host Neighbourhood', 'Host Verifications', 'Street', 'Amenities', 'Features',
    'Host Response Time', 'Host Listings Count', 'Calculated host listings count', 
    'Geolocation', 'Zipcode', 'Host Since', 'Last Scraped', 'Calendar Updated', 
    'Calendar last Scraped', 'First Review', 'Last Review', 'License', 'Square Feet', 
    'Neighbourhood Cleansed', 'Neighbourhood Group Cleansed', 'Review Scores Accuracy',
    'Review Scores Cleanliness','Review Scores Checkin', 'Review Scores Communication',
    'Review Scores Location','Review Scores Value', 'Weekly Price', 'Monthly Price',
    'Availability 60', 'Availability 90' 
])

# Codificacion

le_Neighbourhood = LabelEncoder()
le_property = LabelEncoder()
le_Room = LabelEncoder()
le_bed = LabelEncoder()
le_Cancellation = LabelEncoder()

house_data_train['Neighbourhood'] = le_Neighbourhood.fit_transform(house_data_train['Neighbourhood'])
house_data_train['Property Type'] = le_property.fit_transform(house_data_train['Property Type'])
house_data_train['Room Type'] = le_Room.fit_transform(house_data_train['Room Type'])
house_data_train['Bed Type'] = le_bed.fit_transform(house_data_train['Bed Type'])
house_data_train['Cancellation Policy'] = le_Cancellation.fit_transform(house_data_train['Cancellation Policy'])

# Transformacion: movemos la variable objectivo 'Price' a la primera posición

cols = ['Price'] + [col for col in house_data_train if col != 'Price']
house_data_train = house_data_train[cols]

# Eliminamos outliers

house_data_train = house_data_train[house_data_train['Security Deposit'] <=900.0]
house_data_train = house_data_train[house_data_train['Bathrooms'] <=7]

# Generamos características

house_data_train['bathrooms_pro_bedrooms'] = house_data_train['Bathrooms'] * house_data_train['Bedrooms']
house_data_train['bedrooms_squared'] = house_data_train['Bedrooms'].apply(lambda x: x**2)
house_data_train['beds_per_accommodates'] = house_data_train['Beds'] / house_data_train['Accommodates']


# In[282]:


house_data_train.shape


# Lo mismo para test:

# In[283]:


# Carga de datos de Test
    
house_data_test = pd.read_csv('./data/airbnb_test.csv',sep=';', decimal='.')

# Imputación con loss datos de Train

house_data_test['Neighbourhood'] = house_data_test['Neighbourhood'].fillna(house_data_test['Neighbourhood Cleansed'])
house_data_test['Review Scores Rating'].fillna(house_data_train['Review Scores Rating'].mean(), inplace=True)
house_data_test['Host Response Rate'].fillna(house_data_train['Host Response Rate'].median(), inplace=True)
house_data_test['Host Total Listings Count'].fillna(house_data_train['Host Total Listings Count'].median(), inplace=True)
house_data_test['Price'].fillna(house_data_train['Price'].median(), inplace=True)
house_data_test['Security Deposit'].fillna(house_data_train['Security Deposit'].median(), inplace=True)
house_data_test['Cleaning Fee'].fillna(house_data_train['Cleaning Fee'].median(), inplace=True)
house_data_test['Reviews per Month'].fillna(house_data_train['Reviews per Month'].median(), inplace=True)
house_data_test['Bathrooms'].fillna(house_data_train['Bathrooms'].mode()[0], inplace=True)
house_data_test['Bedrooms'].fillna(house_data_train['Bedrooms'].mode()[0], inplace=True)
house_data_test['Beds'].fillna(house_data_train['Beds'].mode()[0], inplace=True)

# Eliminamos las columnas no importantes segun analisis previo

house_data_test = house_data_test.drop(columns= [
    'ID', 'Scrape ID', 'Host ID','Host Name', 'Name','Host Location',
    'Listing Url', 'Thumbnail Url','Medium Url', 'Picture Url',
    'XL Picture Url', 'Host URL', 'Host Thumbnail Url','Host Picture Url',
    'Experiences Offered', 'Host Acceptance Rate', 'Has Availability',
    'Jurisdiction Names', 'Summary', 'Space', 'Description', 'Neighborhood Overview',
    'Notes', 'Transit', 'Access', 'Interaction', 'House Rules', 'Host About', 
    'Host Neighbourhood', 'Host Verifications', 'Street', 'Amenities', 'Features',
    'Host Response Time', 'Host Listings Count', 'Calculated host listings count', 
    'Geolocation', 'Zipcode', 'Host Since', 'Last Scraped', 'Calendar Updated', 
    'Calendar last Scraped', 'First Review', 'Last Review', 'License', 'Square Feet', 
    'Neighbourhood Cleansed', 'Neighbourhood Group Cleansed', 'Review Scores Accuracy',
    'Review Scores Cleanliness','Review Scores Checkin', 'Review Scores Communication',
    'Review Scores Location','Review Scores Value', 'Weekly Price', 'Monthly Price',
    'Availability 60', 'Availability 90' 
])

# Codificacion

le_Neighbourhood = LabelEncoder()
le_property = LabelEncoder()
le_Room = LabelEncoder()
le_bed = LabelEncoder()
le_Cancellation = LabelEncoder()

house_data_test['Neighbourhood'] = le_Neighbourhood.fit_transform(house_data_test['Neighbourhood'])
house_data_test['Property Type'] = le_property.fit_transform(house_data_test['Property Type'])
house_data_test['Room Type'] = le_Room.fit_transform(house_data_test['Room Type'])
house_data_test['Bed Type'] = le_bed.fit_transform(house_data_test['Bed Type'])
house_data_test['Cancellation Policy'] = le_Cancellation.fit_transform(house_data_test['Cancellation Policy'])

# Transformacion: movemos la variable objectivo 'Price' a la primera posición 

cols = ['Price'] + [col for col in house_data_test if col != 'Price']
house_data_test = house_data_test[cols]

# Eliminamos outliers en Bedrooms

house_data_test = house_data_test[house_data_test['Security Deposit'] <=900.0]
house_data_test = house_data_test[house_data_test['Bathrooms'] <=7]

# Generamos características

house_data_test['bathrooms_pro_bedrooms'] = house_data_test['Bathrooms'] * house_data_test['Bedrooms']
house_data_test['bedrooms_squared'] = house_data_test['Bedrooms'].apply(lambda x: x**2)
house_data_test['beds_per_accommodates'] = house_data_test['Beds'] / house_data_test['Accommodates']


# In[284]:


house_data_test.shape


# Preparamos los datos para sklearn

# In[285]:


from sklearn import preprocessing

# Dataset de train
data_train = house_data_train.values
y_train = data_train[:,0:1]     # nos quedamos con la 1ª columna, price
X_train = data_train[:,1:]      # nos quedamos con el resto

# Dataset de test
data_test = house_data_test.values
y_test = data_test[:,0:1]     # nos quedamos con la 1ª columna, price
X_test = data_test[:,1:]      # nos quedamos con el resto


# In[286]:


print('Datos entrenamiento: ', X_train.shape)
print('Datos test: ', X_test.shape)


# Normalizamos los datos

# In[287]:


# Escalamos (con los datos de train)
scaler = preprocessing.StandardScaler().fit(X_train)

# Transformo train y test con el mismo scaler
XtrainScaled = scaler.transform(X_train)
XtestScaled = scaler.transform(X_test) 


# Primer modelo regresión linear, cross validation con Lasso y revisión de los parámetros

# In[288]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

alpha_vector = np.logspace(-1,10,50)
param_grid = {'alpha': alpha_vector }
grid = GridSearchCV(Lasso(), scoring= 'neg_mean_squared_error', param_grid=param_grid, cv = 3, verbose=2)
grid.fit(X_train, y_train)
print("best mean cross-validation score: {:.3f}".format(grid.best_score_))
print("best parameters: {}".format(grid.best_params_))

#-1 porque es negado
scores = -1*np.array(grid.cv_results_['mean_test_score'])
plt.semilogx(alpha_vector,scores,'-o')
plt.xlabel('alpha',fontsize=16)
plt.ylabel('3-Fold MSE')
plt.show()


# In[289]:


# Con el valor óptimo de alpha corremos lasso para obtener MSE y RMSE

from sklearn.metrics import mean_squared_error

alpha_optimo = grid.best_params_['alpha']
lasso = Lasso(alpha = alpha_optimo).fit(XtrainScaled,y_train)

ytrainLasso = lasso.predict(XtrainScaled)
ytestLasso  = lasso.predict(XtestScaled)
mseTrainModelLasso = mean_squared_error(y_train,ytrainLasso)
mseTestModelLasso = mean_squared_error(y_test,ytestLasso)

print('MSE Modelo Lasso (train): %0.3g' % mseTrainModelLasso)
print('MSE Modelo Lasso (test) : %0.3g' % mseTestModelLasso)

print('RMSE Modelo Lasso (train): %0.3g' % np.sqrt(mseTrainModelLasso))
print('RMSE Modelo Lasso (test) : %0.3g' % np.sqrt(mseTestModelLasso))

feature_names = house_data_train.columns[1:] # es igual en train y en test

w = lasso.coef_
for f,wi in zip(feature_names,w):
    print(f,wi)


# Si nos enfocamos en el resultado del RMSE que esta en la misma unidad que la variable objectivo, su valor (train=46.1/test=41.8) esta muy alto, considerando que la media del precio es 67.43. Necesitamos mejorar los resultados. 

# Vamos a probar reducir características

# Empecemos con el metodo de filtrado para seleccionar características y reducir atributos en nuestro dataset, es decir reducir complejidad

# In[290]:


from sklearn.feature_selection import f_regression, mutual_info_regression

# convertimos el DataFrame al formato necesario para scikit-learn
data = house_data_train.values 

y = data[:,0:1]     # nos quedamos con la 1ª columna, price
X = data[:,1:]      # nos quedamos con el resto

feature_names = house_data_train.columns[1:]

# estudiamos los dos métodos de filtrado
f_test, _ = f_regression(X, y)
f_test /= np.max(f_test)

mi = mutual_info_regression(X, y)
mi /= np.max(mi)

# visualización
plt.figure(figsize=(20, 5))

plt.subplot(1,2,1)
plt.bar(range(X.shape[1]),f_test,  align="center")
plt.xticks(range(X.shape[1]),feature_names, rotation = 90)
plt.xlabel('features')
plt.ylabel('Ranking')
plt.title('$F Test$ score')

plt.subplot(1,2,2)
plt.bar(range(X.shape[1]),mi, align="center")
plt.xticks(range(X.shape[1]),feature_names, rotation = 90)
plt.xlabel('features')
plt.ylabel('Ranking')
plt.title('Mutual information score')

plt.show()


# Podemos inferir algunas variables que se destacan en ambas gráficas (ej: 'Accommodates', 'Cleaning Fee', 'bathrooms_pro_bedrooms', etc), pero hay otras que difieren en nivel en ambos graficos y no es tan concluyente la decision(ej: 'Extra People', 'beds_per_accommodates') .

# También podemos usar la propiedad de los algoritmos basados en árboles para medir la importancia de las variables

# In[291]:


# Arbol de decisión individual

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

maxDepth = range(1,8)
param_grid = {'max_depth': maxDepth }

grid = GridSearchCV(DecisionTreeRegressor(random_state=0), param_grid=param_grid, cv = 5)
grid.fit(X_train, y_train)
print("best mean cross-validation score: {:.3f}".format(grid.best_score_))
print("best parameters: {}".format(grid.best_params_))

scores = np.array(grid.cv_results_['mean_test_score'])
plt.plot(maxDepth,scores,'-o')
plt.xlabel('max_depth',fontsize=16)
plt.ylabel('10-Fold MSE')

plt.show()


# In[292]:


maxDepthOptimo = grid.best_params_['max_depth']
treeModel = DecisionTreeRegressor(max_depth=maxDepthOptimo).fit(X_train,y_train)

print("Train: ",treeModel.score(X_train,y_train))
print("Test: ",treeModel.score(X_test,y_test))


# In[293]:


features = house_data_train.drop(columns=['Price']).columns.tolist()
features_array = np.array(features)

importances = treeModel.feature_importances_
importances = importances / np.max(importances)

indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,10))
plt.barh(range(X_train.shape[1]),importances[indices])
plt.yticks(range(X_train.shape[1]),features_array[indices])
plt.show()


# De acuerdo a la grafica pudieramos seleccionar como variables mas importantes, de 'Maximun Nights' hacia abajo. Una reducción a de 27 a 14 variables. Hagamos una prueba nuevamente regresion lineal 

# In[294]:


# Carga de datos de train con estos ajustes
    
house_data_train = pd.read_csv('./data/airbnb_train.csv',sep=';', decimal='.')

# Imputación con los datos de Train

house_data_train['Price'].fillna(house_data_train['Price'].median(), inplace=True)
house_data_train['Cleaning Fee'].fillna(house_data_train['Cleaning Fee'].median(), inplace=True)
house_data_train['Bathrooms'].fillna(house_data_train['Bathrooms'].mode()[0], inplace=True)
house_data_train['Bedrooms'].fillna(house_data_train['Bedrooms'].mode()[0], inplace=True)
house_data_train['Beds'].fillna(house_data_train['Beds'].mode()[0], inplace=True)
house_data_train['Host Total Listings Count'].fillna(house_data_train['Host Total Listings Count'].median(), inplace=True)

# Generamos características

house_data_train['bathrooms_pro_bedrooms'] = house_data_train['Bathrooms'] * house_data_train['Bedrooms']
house_data_train['beds_per_accommodates'] = house_data_train['Beds'] / house_data_train['Accommodates']

# Codificacion
le_Room = LabelEncoder()
house_data_train['Room Type'] = le_Room.fit_transform(house_data_train['Room Type'])

# Transformacion 

cols = ['Price'] + [col for col in house_data_train if col != 'Price']
house_data_train = house_data_train[cols]

# Eliminamos outliers en Bathroomd

house_data_train = house_data_train[house_data_train['Bathrooms'] <=7]


# In[295]:


# Carga de datos de test con estos ajustes
    
house_data_test = pd.read_csv('./data/airbnb_test.csv',sep=';', decimal='.')

# Imputación con los datos de Train

house_data_test['Price'].fillna(house_data_train['Price'].median(), inplace=True)
house_data_test['Cleaning Fee'].fillna(house_data_train['Cleaning Fee'].median(), inplace=True)
house_data_test['Bathrooms'].fillna(house_data_train['Bathrooms'].mode()[0], inplace=True)
house_data_test['Bedrooms'].fillna(house_data_train['Bedrooms'].mode()[0], inplace=True)
house_data_test['Beds'].fillna(house_data_train['Beds'].mode()[0], inplace=True)
house_data_test['Host Total Listings Count'].fillna(house_data_train['Host Total Listings Count'].median(), inplace=True)

# Generamos características

house_data_test['bathrooms_pro_bedrooms'] = house_data_test['Bathrooms'] * house_data_test['Bedrooms']
house_data_test['beds_per_accommodates'] = house_data_test['Beds'] / house_data_test['Accommodates']

# Codificacion
le_Room = LabelEncoder()
house_data_test['Room Type'] = le_Room.fit_transform(house_data_test['Room Type'])

# Transformacion 

cols = ['Price'] + [col for col in house_data_test if col != 'Price']
house_data_test = house_data_test[cols]

# Eliminamos outliers en Bathrooms

house_data_test = house_data_test[house_data_test['Bathrooms'] <=7]


# In[296]:


# Eliminamos en Train las columnas no importantes, incluyendo las defininidas en la selección de características usando el arbol

house_data_train = house_data_train.drop(columns= [
    'ID', 'Scrape ID', 'Host ID','Host Name', 'Name','Host Location',
    'Listing Url', 'Thumbnail Url','Medium Url', 'Picture Url',
    'XL Picture Url', 'Host URL', 'Host Thumbnail Url','Host Picture Url',
    'Experiences Offered', 'Host Acceptance Rate', 'Has Availability',
    'Jurisdiction Names', 'Summary', 'Space', 'Description', 'Neighborhood Overview',
    'Notes', 'Transit', 'Access', 'Interaction', 'House Rules', 'Host About', 
    'Host Neighbourhood', 'Host Verifications', 'Street', 'Amenities', 'Features',
    'Host Response Time', 'Host Listings Count', 'Calculated host listings count', 
    'Geolocation', 'Zipcode', 'Host Since', 'Last Scraped', 'Calendar Updated', 
    'Calendar last Scraped', 'First Review', 'Last Review', 'License', 'Square Feet', 
    'Neighbourhood Cleansed', 'Neighbourhood Group Cleansed', 'Review Scores Accuracy',
    'Review Scores Cleanliness','Review Scores Checkin', 'Review Scores Communication',
    'Review Scores Location','Review Scores Value', 'Weekly Price', 'Monthly Price',
    'Availability 60', 'Availability 90', 'Host Response Rate', 'Longitude', 
    'Property Type', 'Bed Type', 'Security Deposit', 'Minimum Nights', 'Guests Included',
    'Review Scores Rating', 'Cancellation Policy', 'Reviews per Month','Beds', 'Neighbourhood'
])


# In[297]:


# Eliminamos en Test las columnas no importantes segun selección de características usando el arbol

house_data_test = house_data_test.drop(columns= [
    'ID', 'Scrape ID', 'Host ID','Host Name', 'Name','Host Location',
    'Listing Url', 'Thumbnail Url','Medium Url', 'Picture Url',
    'XL Picture Url', 'Host URL', 'Host Thumbnail Url','Host Picture Url',
    'Experiences Offered', 'Host Acceptance Rate', 'Has Availability',
    'Jurisdiction Names', 'Summary', 'Space', 'Description', 'Neighborhood Overview',
    'Notes', 'Transit', 'Access', 'Interaction', 'House Rules', 'Host About', 
    'Host Neighbourhood', 'Host Verifications', 'Street', 'Amenities', 'Features',
    'Host Response Time', 'Host Listings Count', 'Calculated host listings count', 
    'Geolocation', 'Zipcode', 'Host Since', 'Last Scraped', 'Calendar Updated', 
    'Calendar last Scraped', 'First Review', 'Last Review', 'License', 'Square Feet', 
    'Neighbourhood Cleansed', 'Neighbourhood Group Cleansed', 'Review Scores Accuracy',
    'Review Scores Cleanliness','Review Scores Checkin', 'Review Scores Communication',
    'Review Scores Location','Review Scores Value', 'Weekly Price', 'Monthly Price',
    'Availability 60', 'Availability 90', 'Host Response Rate', 'Longitude', 
    'Property Type', 'Bed Type', 'Security Deposit', 'Minimum Nights', 'Guests Included',
    'Review Scores Rating', 'Cancellation Policy', 'Reviews per Month','Beds', 'Neighbourhood'
])


# In[298]:


print(house_data_train.shape)
print(house_data_test.shape)


# In[299]:


from sklearn import preprocessing

# Dataset de train
data_train = house_data_train.values
y_train = data_train[:,0:1]     # nos quedamos con la 1ª columna, price
X_train = data_train[:,1:]      # nos quedamos con el resto

# Dataset de test
data_test = house_data_test.values
y_test = data_test[:,0:1]     # nos quedamos con la 1ª columna, price
X_test = data_test[:,1:]      # nos quedamos con el resto


# In[300]:


# Escalamos (con los datos de train)
scaler = preprocessing.StandardScaler().fit(X_train)

# Transformo train y test con el mismo scaler
XtrainScaled = scaler.transform(X_train)
XtestScaled = scaler.transform(X_test) 


# In[301]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

alpha_vector = np.logspace(-1,10,50)
param_grid = {'alpha': alpha_vector }
grid = GridSearchCV(Lasso(), scoring= 'neg_mean_squared_error', param_grid=param_grid, cv = 3, verbose=2)
grid.fit(X_train, y_train)
print("best mean cross-validation score: {:.3f}".format(grid.best_score_))
print("best parameters: {}".format(grid.best_params_))

#-1 porque es negado
scores = -1*np.array(grid.cv_results_['mean_test_score'])
plt.semilogx(alpha_vector,scores,'-o')
plt.xlabel('alpha',fontsize=16)
plt.ylabel('3-Fold MSE')
plt.show()


# In[302]:


from sklearn.metrics import mean_squared_error

alpha_optimo = grid.best_params_['alpha']
lasso = Lasso(alpha = alpha_optimo).fit(XtrainScaled,y_train)

ytrainLasso = lasso.predict(XtrainScaled)
ytestLasso  = lasso.predict(XtestScaled)
mseTrainModelLasso = mean_squared_error(y_train,ytrainLasso)
mseTestModelLasso = mean_squared_error(y_test,ytestLasso)

print('MSE Modelo Lasso (train): %0.3g' % mseTrainModelLasso)
print('MSE Modelo Lasso (test) : %0.3g' % mseTestModelLasso)

print('RMSE Modelo Lasso (train): %0.3g' % np.sqrt(mseTrainModelLasso))
print('RMSE Modelo Lasso (test) : %0.3g' % np.sqrt(mseTestModelLasso))

feature_names = house_data_train.columns[1:] # es igual en train y en test

w = lasso.coef_
for f,wi in zip(feature_names,w):
    print(f,wi)


# Aunque simplicamos el modelo con una reduccion a 14 caracteristicas las metricas MSE y RMSE no mejoraron mucho, son muy parecidas
# 

# Probemos otro modelo Random Forest

# In[303]:


# Random Forest

from sklearn.ensemble import RandomForestRegressor

maxDepth = range(1,15)
tuned_parameters = {'max_depth': maxDepth}

grid = GridSearchCV(RandomForestRegressor(random_state=0, n_estimators=200), param_grid=tuned_parameters, cv=5)
grid.fit(X_train, y_train)

print("best mean cross-validation score: {:.3f}".format(grid.best_score_))
print("best parameters: {}".format(grid.best_params_))

scores = np.array(grid.cv_results_['mean_test_score'])
plt.plot(maxDepth,scores,'-o')
plt.xlabel('max_depth')
plt.ylabel('10-fold ACC')

plt.show()


# In[304]:


maxDepthOptimo = grid.best_params_['max_depth']
randomForest = RandomForestRegressor(max_depth=maxDepthOptimo,n_estimators=200,max_features='sqrt').fit(X_train,y_train)

print("Train: ",randomForest.score(X_train,y_train))
print("Test: ",randomForest.score(X_test,y_test))


# Probemos nuevamente Random Forest con las 27 variables iniciales sin la reduccion, para comparar con estos resultados

# In[348]:


# Carga de datos de Train
    
house_data_train = pd.read_csv('./data/airbnb_train.csv',sep=';', decimal='.')

house_data_train['Neighbourhood'] = house_data_train['Neighbourhood'].fillna(house_data_train['Neighbourhood Cleansed'])
house_data_train['Review Scores Rating'].fillna(house_data_train['Review Scores Rating'].mean(), inplace=True)
house_data_train['Host Response Rate'].fillna(house_data_train['Host Response Rate'].median(), inplace=True)
house_data_train['Host Total Listings Count'].fillna(house_data_train['Host Total Listings Count'].median(), inplace=True)
house_data_train['Price'].fillna(house_data_train['Price'].median(), inplace=True)
house_data_train['Security Deposit'].fillna(house_data_train['Security Deposit'].median(), inplace=True)
house_data_train['Cleaning Fee'].fillna(house_data_train['Cleaning Fee'].median(), inplace=True)
house_data_train['Reviews per Month'].fillna(house_data_train['Reviews per Month'].median(), inplace=True)
house_data_train['Bathrooms'].fillna(house_data_train['Bathrooms'].mode()[0], inplace=True)
house_data_train['Bedrooms'].fillna(house_data_train['Bedrooms'].mode()[0], inplace=True)
house_data_train['Beds'].fillna(house_data_train['Beds'].mode()[0], inplace=True)

house_data_train = house_data_train.drop(columns= [
    'ID', 'Scrape ID', 'Host ID','Host Name', 'Name','Host Location',
    'Listing Url', 'Thumbnail Url','Medium Url', 'Picture Url',
    'XL Picture Url', 'Host URL', 'Host Thumbnail Url','Host Picture Url',
    'Experiences Offered', 'Host Acceptance Rate', 'Has Availability',
    'Jurisdiction Names', 'Summary', 'Space', 'Description', 'Neighborhood Overview',
    'Notes', 'Transit', 'Access', 'Interaction', 'House Rules', 'Host About', 
    'Host Neighbourhood', 'Host Verifications', 'Street', 'Amenities', 'Features',
    'Host Response Time', 'Host Listings Count', 'Calculated host listings count', 
    'Geolocation', 'Zipcode', 'Host Since', 'Last Scraped', 'Calendar Updated', 
    'Calendar last Scraped', 'First Review', 'Last Review', 'License', 'Square Feet', 
    'Neighbourhood Cleansed', 'Neighbourhood Group Cleansed', 'Review Scores Accuracy',
    'Review Scores Cleanliness','Review Scores Checkin', 'Review Scores Communication',
    'Review Scores Location','Review Scores Value', 'Weekly Price', 'Monthly Price',
    'Availability 60', 'Availability 90' 
])

le_Neighbourhood = LabelEncoder()
le_property = LabelEncoder()
le_Room = LabelEncoder()
le_bed = LabelEncoder()
le_Cancellation = LabelEncoder()

house_data_train['Neighbourhood'] = le_Neighbourhood.fit_transform(house_data_train['Neighbourhood'])
house_data_train['Property Type'] = le_property.fit_transform(house_data_train['Property Type'])
house_data_train['Room Type'] = le_Room.fit_transform(house_data_train['Room Type'])
house_data_train['Bed Type'] = le_bed.fit_transform(house_data_train['Bed Type'])
house_data_train['Cancellation Policy'] = le_Cancellation.fit_transform(house_data_train['Cancellation Policy'])

cols = ['Price'] + [col for col in house_data_train if col != 'Price']
house_data_train = house_data_train[cols]

house_data_train = house_data_train[house_data_train['Security Deposit'] <=900.0]
house_data_train = house_data_train[house_data_train['Bathrooms'] <=7]

house_data_train['bathrooms_pro_bedrooms'] = house_data_train['Bathrooms'] * house_data_train['Bedrooms']
house_data_train['bedrooms_squared'] = house_data_train['Bedrooms'].apply(lambda x: x**2)
house_data_train['beds_per_accommodates'] = house_data_train['Beds'] / house_data_train['Accommodates']


# In[349]:


# Carga de datos de Test
    
house_data_test = pd.read_csv('./data/airbnb_test.csv',sep=';', decimal='.')

house_data_test['Neighbourhood'] = house_data_test['Neighbourhood'].fillna(house_data_test['Neighbourhood Cleansed'])
house_data_test['Review Scores Rating'].fillna(house_data_train['Review Scores Rating'].mean(), inplace=True)
house_data_test['Host Response Rate'].fillna(house_data_train['Host Response Rate'].median(), inplace=True)
house_data_test['Host Total Listings Count'].fillna(house_data_train['Host Total Listings Count'].median(), inplace=True)
house_data_test['Price'].fillna(house_data_train['Price'].median(), inplace=True)
house_data_test['Security Deposit'].fillna(house_data_train['Security Deposit'].median(), inplace=True)
house_data_test['Cleaning Fee'].fillna(house_data_train['Cleaning Fee'].median(), inplace=True)
house_data_test['Reviews per Month'].fillna(house_data_train['Reviews per Month'].median(), inplace=True)
house_data_test['Bathrooms'].fillna(house_data_train['Bathrooms'].mode()[0], inplace=True)
house_data_test['Bedrooms'].fillna(house_data_train['Bedrooms'].mode()[0], inplace=True)
house_data_test['Beds'].fillna(house_data_train['Beds'].mode()[0], inplace=True)

house_data_test = house_data_test.drop(columns= [
    'ID', 'Scrape ID', 'Host ID','Host Name', 'Name','Host Location',
    'Listing Url', 'Thumbnail Url','Medium Url', 'Picture Url',
    'XL Picture Url', 'Host URL', 'Host Thumbnail Url','Host Picture Url',
    'Experiences Offered', 'Host Acceptance Rate', 'Has Availability',
    'Jurisdiction Names', 'Summary', 'Space', 'Description', 'Neighborhood Overview',
    'Notes', 'Transit', 'Access', 'Interaction', 'House Rules', 'Host About', 
    'Host Neighbourhood', 'Host Verifications', 'Street', 'Amenities', 'Features',
    'Host Response Time', 'Host Listings Count', 'Calculated host listings count', 
    'Geolocation', 'Zipcode', 'Host Since', 'Last Scraped', 'Calendar Updated', 
    'Calendar last Scraped', 'First Review', 'Last Review', 'License', 'Square Feet', 
    'Neighbourhood Cleansed', 'Neighbourhood Group Cleansed', 'Review Scores Accuracy',
    'Review Scores Cleanliness','Review Scores Checkin', 'Review Scores Communication',
    'Review Scores Location','Review Scores Value', 'Weekly Price', 'Monthly Price',
    'Availability 60', 'Availability 90' 
])

le_Neighbourhood = LabelEncoder()
le_property = LabelEncoder()
le_Room = LabelEncoder()
le_bed = LabelEncoder()
le_Cancellation = LabelEncoder()

house_data_test['Neighbourhood'] = le_Neighbourhood.fit_transform(house_data_test['Neighbourhood'])
house_data_test['Property Type'] = le_property.fit_transform(house_data_test['Property Type'])
house_data_test['Room Type'] = le_Room.fit_transform(house_data_test['Room Type'])
house_data_test['Bed Type'] = le_bed.fit_transform(house_data_test['Bed Type'])
house_data_test['Cancellation Policy'] = le_Cancellation.fit_transform(house_data_test['Cancellation Policy'])

cols = ['Price'] + [col for col in house_data_test if col != 'Price']
house_data_test = house_data_test[cols]

house_data_test = house_data_test[house_data_test['Security Deposit'] <=900.0]
house_data_test = house_data_test[house_data_test['Bathrooms'] <=7]

house_data_test['bathrooms_pro_bedrooms'] = house_data_test['Bathrooms'] * house_data_test['Bedrooms']
house_data_test['bedrooms_squared'] = house_data_test['Bedrooms'].apply(lambda x: x**2)
house_data_test['beds_per_accommodates'] = house_data_test['Beds'] / house_data_test['Accommodates']


# In[350]:


print(house_data_train.shape)
print(house_data_test.shape)


# In[308]:


from sklearn import preprocessing

# Dataset de train
data_train = house_data_train.values
y_train = data_train[:,0:1]     # nos quedamos con la 1ª columna, price
X_train = data_train[:,1:]      # nos quedamos con el resto

# Dataset de test
data_test = house_data_test.values
y_test = data_test[:,0:1]     # nos quedamos con la 1ª columna, price
X_test = data_test[:,1:]      # nos quedamos con el resto


# In[351]:


# Escalamos (con los datos de train)
scaler = preprocessing.StandardScaler().fit(X_train)

# Transformo train y test con el mismo scaler
XtrainScaled = scaler.transform(X_train)
XtestScaled = scaler.transform(X_test) 


# In[352]:


# Random Forest 2 corrida

from sklearn.ensemble import RandomForestRegressor

maxDepth = range(1,15)
tuned_parameters = {'max_depth': maxDepth}

grid = GridSearchCV(RandomForestRegressor(random_state=0, n_estimators=200), param_grid=tuned_parameters, cv=5)
grid.fit(X_train, y_train)

print("best mean cross-validation score: {:.3f}".format(grid.best_score_))
print("best parameters: {}".format(grid.best_params_))

scores = np.array(grid.cv_results_['mean_test_score'])
plt.plot(maxDepth,scores,'-o')
plt.xlabel('max_depth')
plt.ylabel('10-fold ACC')

plt.show()


# In[353]:


maxDepthOptimo = grid.best_params_['max_depth']
randomForest = RandomForestRegressor(max_depth=maxDepthOptimo,n_estimators=200,max_features='sqrt').fit(X_train,y_train)

print("Train: ",randomForest.score(X_train,y_train))
print("Test: ",randomForest.score(X_test,y_test))


# Mejores prestaciones que el anterior. Veamos la importancia de las caracteristicas con Random Forest

# In[354]:


features = house_data_train.drop(columns=['Price']).columns.tolist()
features_array = np.array(features)

importances = randomForest.feature_importances_
importances = importances / np.max(importances)

indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,10))
plt.barh(range(X_train.shape[1]),importances[indices])
plt.yticks(range(X_train.shape[1]),features_array[indices])
plt.show()


# Sin duda podemos mejorar los resultados, quizas con regularización o seleccionando un combinación diferente de caracteristicas con otro metodo. Sin embargo corramos otros modelos como Bagging, Gradient Boosting y SVR para comparar resultados

# In[355]:


# Modelo con Bagging

from sklearn.ensemble import BaggingRegressor

maxDepth = range(1,15)
tuned_parameters = {'base_estimator__max_depth': maxDepth}

grid = GridSearchCV(BaggingRegressor(estimator=DecisionTreeRegressor(), random_state=0, n_estimators=200), param_grid=tuned_parameters, cv=5)
grid.fit(X_train, y_train)

print("best mean cross-validation score: {:.3f}".format(grid.best_score_))
print("best parameters: {}".format(grid.best_params_))

scores = np.array(grid.cv_results_['mean_test_score'])
plt.plot(maxDepth,scores,'-o')
plt.xlabel('max_depth')
plt.ylabel('10-fold ACC')

plt.show()


# In[356]:


maxDepthOptimo = grid.best_params_['base_estimator__max_depth']
baggingModel = BaggingRegressor(base_estimator=DecisionTreeRegressor(max_depth=maxDepthOptimo),n_estimators=200).fit(X_train,y_train)

print("Train: ",baggingModel.score(X_train,y_train))
print("Test: ",baggingModel.score(X_test,y_test))


# In[357]:


# Gradient Boosting

from sklearn.ensemble import GradientBoostingRegressor

Niterations = [500,1000,1500,2000]
learningRate = [0.1,0.05]

param_grid = {'n_estimators': Niterations,'learning_rate':learningRate }
grid = GridSearchCV(GradientBoostingRegressor(random_state=0, max_depth=3), param_grid=param_grid, cv = 3, verbose=2)
grid.fit(X_train, y_train)
print("best mean cross-validation score: {:.3f}".format(grid.best_score_))
print("best parameters: {}".format(grid.best_params_))


# In[358]:


lrOptimo = grid.best_params_['learning_rate']
neOptimo = grid.best_params_['n_estimators']
bt = GradientBoostingRegressor(random_state=0, max_depth=3,learning_rate=lrOptimo, n_estimators=neOptimo)
bt.fit(X_train,y_train)

error = 1-grid.cv_results_['mean_test_score'].reshape(len(learningRate),len(Niterations))
colors = ['r','b','g','k','m']
for i,lr in enumerate(learningRate):    
    plt.plot(Niterations,error[i,:],colors[i] + '--o',label='lr = %g'%lr)

plt.legend()
plt.xlabel('# iteraciones')
plt.ylabel('5-fold CV Error')
plt.title('train: %0.3f\ntest:  %0.3f'%(bt.score(X_train,y_train),bt.score(X_test,y_test)))
plt.grid()
plt.show()


# In[359]:


lrOptimo = grid.best_params_['learning_rate']
neOptimo = grid.best_params_['n_estimators']
baggingModel = GradientBoostingRegressor(max_depth=3, n_estimators=neOptimo, learning_rate=lrOptimo).fit(X_train,y_train)

print("Train: ",baggingModel.score(X_train,y_train))
print("Test: ",baggingModel.score(X_test,y_test))


# In[360]:


# SVM

from sklearn.svm import SVR

# Paso 2:
vectorC = np.logspace(-2, 2, 10)
vectorG = np.logspace(-5, 1, 4)

param_grid = {'C': vectorC, 'gamma':vectorG}
grid = GridSearchCV(SVR(kernel='rbf'), param_grid=param_grid, cv = 5, verbose=2)
grid.fit(XtrainScaled, y_train)


# In[361]:


print("best mean cross-validation score: {:.3f}".format(grid.best_score_))
print("best parameters: {}".format(grid.best_params_))

print("Gamma en la gráfica: ", np.log10(grid.best_params_['gamma']))
print("C en la gráfica: ", np.log10(grid.best_params_['C']))

# Mostramos prestaciones en CV
scores = grid.cv_results_['mean_test_score'].reshape(len(vectorC),len(vectorG))

plt.figure(figsize=(10,6))
plt.imshow(scores, interpolation='nearest', vmin= 0.6, vmax=0.9)
plt.xlabel('log(gamma)')
plt.ylabel('log(C)')
plt.colorbar()
plt.xticks(np.arange(len(vectorG)), np.log10(vectorG), rotation=90)
plt.yticks(np.arange(len(vectorC)), np.log10(vectorC))
plt.title('5-fold accuracy')
plt.show()


# In[362]:


# Paso 3:
Copt = grid.best_params_['C']
Gopt = grid.best_params_['gamma']

svmModel = SVR(kernel='rbf',gamma = Gopt, C = Copt).fit(XtrainScaled,y_train)
print(f'Acc (TEST): {svmModel.score(XtestScaled,y_test):0.2f}')


# Resumen de modelos aplicados y metricas resultantes

# TreeModel 
# Train:  0.5513480927203134
# Test:  0.4812121163461571
# 
# RandomForest
# Train:  0.8990836046233255
# Test:  0.6395901440232641
# 
# Bagging
# Train:  0.9191330025709301
# Test:  0.6068611150157666
# 
# Gradient Boosting
# Train:  0.7570271569990576
# Test:  0.6097171001330846
# 
# SVM
# Acc (Test): 0.52 
# 
# Aunque esperaba iterar mas y conseguir mejores prestaciones, a continuación mi conclusión final del estudio.
# 
# De acuerdo a los resultados obtenidos el mejor modelo resulta ser Randon Forest, al ofrecer una mejor metrica de accurancy (sobre todo en Test) para predecir el precio del inmueble. 
# 
# Random Forest ayuda a simplificar el modelo al ser mas acertivo (que otros metodos) en la seleccion de las caracteristicas relevantes o predictoras del precio, asi como también ofrece una mejor generalizacion de los datos reduciendo el overfitting.  
