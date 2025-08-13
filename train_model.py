# train_model.py
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pickle
import pandas as pd

iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names  # ['sepal length (cm)', ...]

# Optional: use DataFrame for clarity
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# stratify ensures class balance in splits
X_train, X_test, y_train, y_test = train_test_split(
    df[feature_names], y, test_size=0.2, random_state=42 
)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Create a LabelEncoder that maps 0/1/2 -> 'setosa'/'versicolor'/'virginica'
le = LabelEncoder()
le.fit(iris.target_names)   # fit on ['setosa','versicolor','virginica']
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("Saved model.pkl and label_encoder.pkl")
