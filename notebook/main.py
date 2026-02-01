# ================================
# IMPORT REQUIRED LIBRARIES
# ================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set default plot style
plt.style.use("default")


# ================================
# LOAD DATASET
# ================================
# Read CSV file from data folder
df = pd.read_csv("data/StudentsPerformance.csv")

# Display first 5 rows
df.head()


# ================================
# DATA UNDERSTANDING
# ================================
# Check dataset shape
df.shape

# Check data types and null values
df.info()

# Statistical summary
df.describe()

# Count missing values
df.isnull().sum()


# ================================
# DATA CLEANING
# ================================
# Rename column names to lowercase and remove spaces
df.columns = df.columns.str.replace(" ", "_").str.lower()

# Check updated columns
df.head()


# ================================
# FEATURE ENGINEERING
# ================================
# Create total score column
df["total_score"] = df["math_score"] + df["reading_score"] + df["writing_score"]

# Create average score column
df["average_score"] = df["total_score"] / 3


# ================================
# DROPOUT RISK LABEL CREATION
# ================================
# Function to classify dropout risk
def risk_level(avg):
    if avg < 40:
        return "High Risk"
    elif avg < 60:
        return "Medium Risk"
    else:
        return "Low Risk"

# Apply risk classification
df["dropout_risk"] = df["average_score"].apply(risk_level)

df.head()


# ================================
# EXPLORATORY DATA ANALYSIS (EDA)
# ================================

# Gender vs Average Score
plt.figure(figsize=(6,4))
sns.boxplot(x="gender", y="average_score", data=df)
plt.title("Gender vs Average Score")
plt.savefig("visuals/gender_vs_average_score.png", bbox_inches="tight")
plt.show()


# ================================
# Parental Education Impact
# ================================
plt.figure(figsize=(10,5))
sns.barplot(x="parental_level_of_education", y="average_score", data=df)
plt.xticks(rotation=45)
plt.title("Parental Education vs Performance")
plt.savefig("visuals/parental_education_vs_performance.png", bbox_inches="tight")
plt.show()


# ================================
# Lunch Type Impact
# ================================
plt.figure(figsize=(6,4))
sns.boxplot(x="lunch", y="average_score", data=df)
plt.title("Lunch Type vs Performance")
plt.savefig("visuals/lunch_type_vs_performance.png", bbox_inches="tight")
plt.show()


# ================================
# Test Preparation Course Effect
# ================================
plt.figure(figsize=(6,4))
sns.boxplot(x="test_preparation_course", y="average_score", data=df)
plt.title("Test Preparation Course vs Performance")
plt.savefig("visuals/test_preparation_vs_performance.png", bbox_inches="tight")
plt.show()


# ================================
# CORRELATION ANALYSIS
# ================================
plt.figure(figsize=(6,4))
sns.heatmap(
    df[["math_score", "reading_score", "writing_score", "average_score"]].corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Between Scores")
plt.savefig("visuals/score_correlation_heatmap.png", bbox_inches="tight")
plt.show()


# ================================
# DROPOUT RISK DISTRIBUTION
# ================================
plt.figure(figsize=(6,4))
sns.countplot(x="dropout_risk", data=df)
plt.title("Dropout Risk Distribution")
plt.savefig("visuals/dropout_risk_distribution.png", bbox_inches="tight")
plt.show()


# ================================
# SAVE CLEANED DATA
# ================================
# Save processed dataset
df.to_csv("cleaned_student_data.csv", index=False)
