import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler  # ✅ Correct Import
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.stats import skew, kurtosis
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=';')
plt.figure(figsize=(8, 6))
sns.scatterplot(x="alcohol", y="quality", data=df)
plt.xlabel("Alcohol Content (%)")
plt.ylabel("Wine Quality Score")
plt.title("Alcohol vs Wine Quality")
plt.show()
plt.figure(figsize=(8, 6))
sns.countplot(x="quality", data=df, palette="muted")
plt.xlabel("Wine Quality")
plt.ylabel("Count")
plt.title("Distribution of Wine Quality")
plt.show()
plt.figure(figsize=(10, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Wine Features")
plt.show()
alcohol_data = df["alcohol"].dropna()
print(f"Statistical Moments for Alcohol Content:")
print(f"Mean: {np.mean(alcohol_data):.2f}, Variance: {np.var(alcohol_data):.2f}")
print(f"Skewness: {skew(alcohol_data):.2f}, Kurtosis: {kurtosis(alcohol_data):.2f}")
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[["alcohol", "volatile acidity"]])
kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(df_scaled)
plt.figure(figsize=(8, 6))
sns.scatterplot(x="alcohol", y="volatile acidity", hue=df["Cluster"], palette="viridis", data=df)
plt.xlabel("Alcohol Content (%)")
plt.ylabel("Volatile Acidity")
plt.title("K-Means Clustering of Wine based on Alcohol and Acidity")
plt.legend(title="Cluster")
plt.show()
X = df[["alcohol"]]
y = df["quality"]
poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
poly_model.fit(X, y)
plt.figure(figsize=(8, 6))
sns.scatterplot(x="alcohol", y="quality", data=df, label="Data")
x_range = np.linspace(df["alcohol"].min(), df["alcohol"].max(), 100).reshape(-1, 1)
plt.plot(x_range, poly_model.predict(x_range), color="red", label="Polynomial Fit")
plt.xlabel("Alcohol Content (%)")
plt.ylabel("Wine Quality Score")
plt.title("Polynomial Regression Fit: Alcohol vs Wine Quality")
plt.legend()
plt.show()
