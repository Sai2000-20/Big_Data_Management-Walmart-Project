#PySpark Code for Preprocessing (Categorization and max sales store) and Processing (Prediction) by Saira

#pip3 install PyArrow
#pip3 install statsmodels
#pip3 install numpy
#pip3 install pandas
# import the necessary pyspark and pandas libraries

from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import StructType,StructField,StringType,LongType,DoubleType,FloatType
import statsmodels.tsa.api as sm
import numpy as np
import pandas as pd
from pyspark.sql.functions import col, expr, when
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import col
import pyspark.sql.functions as sf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

currentTime=time.time()

# read the entire data as spark dataframe from local harddisk
walmart_data = spark.read.format('csv').options(header='true', inferSchema='true').load('/home/ubuntu/Downloads/walmart.csv')\
.select('Store','Dept','Date','Weekly_Sales','Size')

walmart_data.show()

executionTime=time.time()-currentTime
print(executionTime)
currentTime=time.time()
#2.037343978881836
walmart_data1=walmart_data.withColumn("category",
       when((col("Size") >=0) & (col("Size") <= 100000 ), "Small")
      .when((col("Size") >100000) & (col("Size") <= 200000 ), "Medium")
      .when((col("Size") >200000) &(col("Size") <= 300000 ), "Large")
      .otherwise("Na"))#.show()

executionTime=time.time()-currentTime
print(executionTime)
currentTime=time.time()
 #0.21456623077392578
 
from pyspark.sql.functions import col, countDistinct

df_count=walmart_data1.groupBy("category").agg(*(countDistinct(col(Store)).alias(Store) for Store in walmart_data1.columns))
df_count.select(df_count.columns[0:2]).show()
df_count=df_count['category','Store']
#walmart_data1.show()
x=['Medium','Small','Large']
y=[18,14,13]
plt.bar(x,y)
plt.xlabel('Store Category')
plt.ylabel('Store Count')
plt.title('Count of Store each category')
executionTime=time.time()-currentTime
print(executionTime)
currentTime=time.time()
#1.4242165088653564
plt.show()
# |category|Store|Dept|Date|Weekly_Sales|Size|category|
# +--------+-----+----+----+------------+----+--------+
# |  Medium|   18|  82| 182|      161793|  18|       1|
# |   Small|   14|  78| 182|       95262|  10|       1|
# |   Large|   13|  81| 182|      124581|  12|       1|
# +--------+-----+----+----+------------+----+--------+

#walmart_data1.groupBy("category").pisot("category").count().show()
df_temp=walmart_data1['category','Store']
df_temp=df_temp.select('category','Store').distinct()
df_temp.filter("category == 'Large'").show()
df_temp.filter("category == 'Medium'").show()
df_temp.filter("category == 'Small'").show()


walmart_data1.filter("category == 'Large'").show()
walmart_data1.filter("Size >=200000").show()

#Cleaning Data with null Weekly_Sales
walmart_data1.filter("Weekly_Sales == 'NA'").show()

walmart_data1 = walmart_data1.filter("Weekly_Sales != 'NA'")#.show()
walmart_data1.filter("Weekly_Sales == 'NA'").show()

executionTime=time.time()-currentTime
print(executionTime)
currentTime=time.time()

#Get Maximum Sum of sales per category
import pyspark.sql.functions as sf
walmart_data2=  walmart_data1.select(walmart_data1.Weekly_Sales.cast("float"),walmart_data1.Store,walmart_data1.Dept,walmart_data1.Size,walmart_data1.category,walmart_data1.Date)
walmart_data_sum = walmart_data2.groupBy(['Store','category']).agg(sf.sum("Weekly_Sales").alias('Weekly_Sales_sum'))
#walmart_data_sum.show()
walmart_data_max=walmart_data_sum.join(walmart_data_sum.groupBy('category').agg(sf.max('Weekly_Sales_sum').alias('Weekly_Sales_sum')),on='Weekly_Sales_sum',how='leftsemi')#.show()
walmart_data_max.show()
executionTime=time.time()-currentTime
print(executionTime)
#7.101712942123413
currentTime=time.time()

sum_y = list( walmart_data_max.select('Weekly_Sales_sum').toPandas()['Weekly_Sales_sum'])
sum_x = list( walmart_data_max.select('category').toPandas()['category'])


##PLOT Stores by category
large_y = list( walmart_data_sum.filter("category == 'Large'").select('Weekly_Sales_sum').toPandas()['Weekly_Sales_sum'])
large_x = list( walmart_data_sum.filter("category == 'Large'").select('Store').toPandas()['Store'])
#list_string = list(map(str, large_x))
#plt.xticks(list_string)
plt.bar(large_x,large_y)
plt.show()



plt.xticks(range(0,len(sum_x)), sum_x)
#plt.show()
plt.xlabel('Store Category')
plt.ylabel('Sales X100 Million USD')
plt.title('Sales of Top performing Stores')
plt.bar(sum_x,sum_y)
plt.show()

#########################FUTURE PREDICTION of Sales for 3 months ###########

## basic data cleaning before implementing the pandas udf
##removing Store - Dept combination with less than 2 years (52 weeks ) of data : 2 Years = 104.28 Weeks
# Group by Store and Size Category
executionTime=time.time()-currentTime
print(executionTime)
currentTime=time.time()

walmart_filter_data = walmart_data2.groupBy(['Store','Dept']).count().filter("count >= 104").select("Store","Dept")
#join data for Store, Dept 
walmart_data_selected_store_departments = walmart_data2.join(walmart_filter_data,['Store','Dept'],'inner')
walmart_data_selected_store_departments.show()

#JOIN Cleaned Data with Store with Manimum Sales as per category
cleaned_filter_data_for_prediction = walmart_data_max.join(walmart_data_selected_store_departments,['Store','category'],'inner')
cleaned_filter_data_for_prediction.show()

executionTime=time.time()-currentTime
print(executionTime)
#9.788768768310547
currentTime=time.time()

##Define pandas Output data frame

data_schema = StructType([StructField('Store', StringType(), True),
                     StructField('category', StringType(), True),
                     StructField('weekly_forecast_1', DoubleType(), True),
                     StructField('weekly_forecast_2', DoubleType(), True),
                     StructField('weekly_forecast_3', DoubleType(), True),
                     StructField('weekly_forecast_4', DoubleType(), True),
                     StructField('weekly_forecast_5', DoubleType(), True),
                     StructField('weekly_forecast_6', DoubleType(), True),
                     StructField('weekly_forecast_7', DoubleType(), True),
                     StructField('weekly_forecast_8', DoubleType(), True),
                     StructField('weekly_forecast_9', DoubleType(), True),
                     StructField('weekly_forecast_10', DoubleType(), True),
                     StructField('weekly_forecast_11', DoubleType(), True),
                     StructField('weekly_forecast_12', DoubleType(), True),
                     StructField('weekly_forecast_13', DoubleType(), True)])

#Using User Defined Function for Prediction
@pandas_udf(data_schema, PandasUDFType.GROUPED_MAP)
def time_series_udf(data):
    data.set_index('Date',inplace = True)
    time_series_data = data['Weekly_Sales']
    ##the model
    model_monthly = sm.ExponentialSmoothing(np.asarray(time_series_data),trend='add').fit()
    ##forecast values
    forecast_values = pd.Series(model_monthly.forecast(13),name = 'fitted_values')
    #forecast_values.show()
    return pd.DataFrame({'Store': [str(data.Store.iloc[0])],'category': [str(data.category.iloc[0])]     ,'weekly_forecast_1': [forecast_values[0]], 'weekly_forecast_2':[forecast_values[1]],'weekly_forecast_3': [forecast_values[2]],'weekly_forecast_4': [forecast_values[3]],'weekly_forecast_5': [forecast_values[4]],'weekly_forecast_6': [forecast_values[5]],'weekly_forecast_7': [forecast_values[6]],'weekly_forecast_8': [forecast_values[7]],'weekly_forecast_9': [forecast_values[8]],'weekly_forecast_10': [forecast_values[9]],'weekly_forecast_11': [forecast_values[10]],'weekly_forecast_12': [forecast_values[11]],'weekly_forecast_13': [forecast_values[12]] })


##aggregating the forecasted results in the form of a spark dataframe

executionTime=time.time()-currentTime
print(executionTime)
currentTime=time.time()

predicted_spark_df = cleaned_filter_data_for_prediction.groupby(['Store','category']).apply(time_series_udf)
## to see the forecasted results
predicted_spark_df.show(10)
executionTime=time.time()-currentTime
print(executionTime)
#12.319074630737305
currentTime=time.time()

#Prediction_store_max_sales = walmart_data_max.join(predicted_spark_df,['Store','category'],'inner')

#20:large,17:small,10:medium
walmart_data.filter("category == 'Large'")
predicted_spark_df.filter("Store='20'").show()

predicted_spark_df.filter("Store='20'")['weekly_forecast_1','weekly_forecast_2'].show()


store_large=predicted_spark_df.filter("Store='20'")['weekly_forecast_1','weekly_forecast_2','weekly_forecast_3','weekly_forecast_4','weekly_forecast_5','weekly_forecast_6','weekly_forecast_7','weekly_forecast_8','weekly_forecast_9','weekly_forecast_10','weekly_forecast_11','weekly_forecast_12','weekly_forecast_13'].collect()
store_small=predicted_spark_df.filter("Store='17'")['weekly_forecast_1','weekly_forecast_2','weekly_forecast_3','weekly_forecast_4','weekly_forecast_5','weekly_forecast_6','weekly_forecast_7','weekly_forecast_8','weekly_forecast_9','weekly_forecast_10','weekly_forecast_11','weekly_forecast_12','weekly_forecast_13'].collect()
store_medium=predicted_spark_df.filter("Store='10'")['weekly_forecast_1','weekly_forecast_2','weekly_forecast_3','weekly_forecast_4','weekly_forecast_5','weekly_forecast_6','weekly_forecast_7','weekly_forecast_8','weekly_forecast_9','weekly_forecast_10','weekly_forecast_11','weekly_forecast_12','weekly_forecast_13'].collect()

from datetime import datetime
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DateType

x = list(range(1,13+1))
plt.plot(x,store_large[0], marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4, label="Category Large(#20)")
plt.plot(x,store_small[0], marker='x', color='olive', linewidth=2, label="CategorySmall(#17)")
plt.plot(x,store_medium[0], marker='+', color='olive', linewidth=2, linestyle='dashed', label="CategoryMedium(#10)")
plt.legend()
plt.xlabel('Future Week Numbers')
plt.ylabel('Weekly Sales in USD')
plt.title('Predicted Weekly Sales of Top performing Stores per category since 2012-10-26')
plt.show()

# multiple line plot

