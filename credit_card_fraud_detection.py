import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load data
data = pd.read_csv("fraudTrain.csv")

# Select useful columns
data = data[['amt', 'category', 'gender', 'city_pop', 'is_fraud']]

# Convert categorical columns to numeric
le = LabelEncoder()
data['category'] = le.fit_transform(data['category'])
data['gender'] = le.fit_transform(data['gender'])

# Split features and target
X = data.drop('is_fraud', axis=1)
y = data['is_fraud']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
