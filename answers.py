# ============================================
# Assignment: Data Analysis with Pandas & Matplotlib
# ============================================

# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# Task 1: Load and Explore the Dataset

try:
    # Loading dataset Iris dataset from seaborn 
    df = sns.load_dataset("iris")  
    
    print("✅ Dataset loaded successfully!\n")
    
except FileNotFoundError:
    print("❌ Error: Dataset not found. Please check the file path.")
except Exception as e:
    print("❌ An unexpected error occurred:", e)

# Displaying first few rows
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Checking data types & missing values
print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# Cleaning dataset (droping rows with missing values if any)
df = df.dropna()


# Task 2: Basic Data Analysis

# Basic statistics
print("\nBasic Statistics:")
print(df.describe())

# Grouping: average petal_length per species
grouped = df.groupby("species")["petal_length"].mean()
print("\nAverage Petal Length per Species:")
print(grouped)

# Simple observation
print("\nObservation: Setosa species has the shortest average petal length, while Virginica has the longest.")

# Task 3: Data Visualization

# 1. Line chart (trend example - use index as pseudo time)
plt.figure(figsize=(8,5))
plt.plot(df.index, df["sepal_length"], label="Sepal Length", color="blue")
plt.title("Line Chart: Sepal Length Trend")
plt.xlabel("Index")
plt.ylabel("Sepal Length")
plt.legend()
plt.show()

# 2. Bar chart (avg petal length per species)
plt.figure(figsize=(8,5))
sns.barplot(x="species", y="petal_length", data=df, palette="viridis")
plt.title("Bar Chart: Avg Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length")
plt.show()

# 3. Histogram (distribution of sepal length)
plt.figure(figsize=(8,5))
plt.hist(df["sepal_length"], bins=20, color="purple", edgecolor="black")
plt.title("Histogram: Distribution of Sepal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter plot (sepal length vs petal length)
plt.figure(figsize=(8,5))
sns.scatterplot(x="sepal_length", y="petal_length", hue="species", data=df, palette="Set2")
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.legend(title="Species")
plt.show()
