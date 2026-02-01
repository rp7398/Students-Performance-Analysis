"""
**IMPORT LIBRARIES**
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("default")

"""**LOAD DATASET**"""

df = pd.read_csv("data/StudentsPerformance.csv")
df.head()

"""**DATA UNDERSTANDING**"""

df.shape

df.info()

df.describe()

df.isnull().sum()

"""**DATA CLEANING**"""

# Rename columns for simplicity
df.columns = df.columns.str.replace(" ", "_").str.lower()

df.head()

"""**FEATURE ENGINEERING**"""

df["total_score"] = df["math_score"] + df["reading_score"] + df["writing_score"]
df["average_score"] = df["total_score"] / 3

"""**Dropout Risk Label**"""

def risk_level(avg):
    if avg < 40:
        return "High Risk"
    elif avg < 60:
        return "Medium Risk"
    else:
        return "Low Risk"

df["dropout_risk"] = df["average_score"].apply(risk_level)
df.head()

"""**EXPLORATORY DATA ANALYSIS (EDA)**"""

plt.figure(figsize=(6,4))
sns.boxplot(x="gender", y="average_score", data=df)
plt.title("Gender vs Average Score")

plt.savefig("visuals/gender_vs_average_score.png", bbox_inches="tight")
plt.show()

"""**Parental Education Impact**"""

plt.figure(figsize=(10,5))
sns.barplot(x="parental_level_of_education", y="average_score", data=df)
plt.xticks(rotation=45)
plt.title("Parental Education vs Performance")

plt.savefig("visuals/parental_education_vs_performance.png", bbox_inches="tight")
plt.show()

"""## **Lunch Type Impact**"""

plt.figure(figsize=(6,4))
sns.boxplot(x="lunch", y="average_score", data=df)
plt.title("Lunch Type vs Performance")

plt.savefig("visuals/lunch_type_vs_performance.png", bbox_inches="tight")
plt.show()

"""**Test Preparation Effect**"""

plt.figure(figsize=(6,4))
sns.boxplot(x="test_preparation_course", y="average_score", data=df)
plt.title("Test Preparation Course vs Performance")

plt.savefig("visuals/test_preparation_vs_performance.png", bbox_inches="tight")
plt.show()

"""**CORRELATION ANALYSIS**"""

plt.figure(figsize=(6,4))
sns.heatmap(
    df[["math_score", "reading_score", "writing_score", "average_score"]].corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Between Scores")

plt.savefig("visuals/score_correlation_heatmap.png", bbox_inches="tight")
plt.show()

"""**DROPOUT RISK ANALYSIS**"""

plt.figure(figsize=(6,4))
sns.countplot(x="dropout_risk", data=df)
plt.title("Dropout Risk Distribution")

plt.savefig("visuals/dropout_risk_distribution.png", bbox_inches="tight")
plt.show()

"""**SAVE CLEANED DATA**"""

df.to_csv("cleaned_student_data.csv", index=False)