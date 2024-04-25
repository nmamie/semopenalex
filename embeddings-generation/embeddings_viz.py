import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import PCA
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import array_to_vector
from pyspark.sql.functions import col, expr
from pyspark.sql import SparkSession



spark = SparkSession.builder.appName('embeddings_viz').config('spark.driver.memory', '800g').config('spark.executor.memory', '800g').getOrCreate()
# Load the data
df = spark.read.parquet('embeddings.parquet')

print("Number of rows: ", df.count())
print("Number of columns: ", len(df.columns))
df.show(1, truncate=False)
print("DATASET LOAD DONE")


## Step 1: Convert array<float> to linalg.Vector
df_vectorized = df.withColumn("embedding", array_to_vector(col("embedding")))

# print first row
df_vectorized.show(1, truncate=False)
print("VECTORIZE DONE")

# Step 2: Apply PCA
pca = PCA(k=2, inputCol="embedding", outputCol="pcaFeatures")
model = pca.fit(df_vectorized)
result = model.transform(df_vectorized).select("pcaFeatures")

# print first row
result.show(1, truncate=False)
print(model.explainedVariance)
print("PCA DONE")

# Step 3 save the PCA features to a parquet file
result.write.parquet("pca_features.parquet")
print("PCA FEATURES PARQUET FILE SAVED")

# Step 4: visualize the PCA features
pdf = result.toPandas()
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(pdf["pca_features"].apply(lambda x: x[0]), pdf["pca_features"].apply(lambda x: x[1]))
ax.set_xlabel("PCA Feature 1")
ax.set_ylabel("PCA Feature 2")
ax.set_title("PCA Features")
# save the plot
plt.savefig("pca_features.png")
print("PCA FEATURES PLOT SAVED")