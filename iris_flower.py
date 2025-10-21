
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load dataset
df = pd.read_csv("Downloads/Iris.csv")

# 2. Prepare features (X) and labels (y)
X = df.drop(columns=["Id", "Species"])
y = df["Species"]

# 3. Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 7. visualize confusion matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues",
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# 8. visualize feature distributions
df_melted = pd.melt(df, id_vars=["Species"], value_vars=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
                    var_name="Feature", value_name="Value")
plt.figure(figsize=(10, 6))
sns.boxplot(x="Feature", y="Value", hue="Species", data=df_melted)
plt.title("Feature Distributions by Species")   
plt.show()  


