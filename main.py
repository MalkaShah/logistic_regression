#type:ignore
# Standard operational package imports.
import numpy as np
import pandas as pd
import pprint

# Important imports for preprocessing, modeling, and evaluation.
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix

# Visualization package imports.
import matplotlib.pyplot as plt
import seaborn as sns

# Loading Data
df_original = pd.read_csv("Invistico_Airline.csv")
print(df_original.head(n = 10))

# EDA
print(df_original.dtypes)
print(df_original['satisfaction'].value_counts(dropna = False)) #to check number of satisfies customers
print(df_original.isnull().sum()) # to check for null values
df_subset = df_original.dropna(axis=0).reset_index(drop = True) #as 300+ values in arrival time so missing values are removed
"""to create a (sns.regplot) of model to visualize results later the independent variable Inflight entertainment cannot be "of type int" and the dependent variable satisfaction cannot be "of type object. 
# Making the Inflight entertainment column "of type float. and Convert the categorical column satisfaction into numeric through one-hot encoding."""
df_subset = df_subset.astype({"Inflight entertainment": float})
df_subset['satisfaction'] = OneHotEncoder(drop='first').fit_transform(df_subset[['satisfaction']]).toarray()

pprint.pprint(df_subset.head(10))


# Training data
X = df_subset[["Inflight entertainment"]]
y = df_subset["satisfaction"]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)


# Model Training
clf = LogisticRegression().fit(X_train,y_train)
print("Coeffient:",clf.coef_)
print("Intercept",clf.intercept_)

# Result
sns.regplot(x="Inflight entertainment", y="satisfaction", data=df_subset, logistic=True, ci=None)
plt.show()
y_pred = clf.predict(X_test)
clf.predict_proba(X_test)
print("Accuracy:", "%.6f" % metrics.accuracy_score(y_test, y_pred))
print("Precision:", "%.6f" % metrics.precision_score(y_test, y_pred))
print("Recall:", "%.6f" % metrics.recall_score(y_test, y_pred))
print("F1 Score:", "%.6f" % metrics.f1_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels = clf.classes_)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.show()