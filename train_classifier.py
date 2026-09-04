import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("=" * 60)
print("  ASL Recognition Model Training & Evaluation")
print("=" * 60)

# 1. Load extracted landmarks
data_path = './data.pickle'
try:
    data_dict = pickle.load(open(data_path, 'rb'))
except FileNotFoundError:
    print(f"Error: '{data_path}' not found. Please run create_dataset.py first.")
    exit(1)

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])
classes = np.unique(labels)

print(f"Total samples: {len(data)}")
print(f"Total classes ({len(classes)}): {list(classes)}")
print(f"Feature dimension: {data.shape[1]}")

# 2. Split into 80% training data, 20% testing data
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
)

# 3. Train the Random Forest Model
print("\nTraining RandomForestClassifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(x_train, y_train)

# 4. Test Accuracy and Metrics
y_predict = model.predict(x_test)
score = accuracy_score(y_test, y_predict)
print(f"\nModel Training Complete!")
print(f"Test Accuracy: {score * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_predict, zero_division=0))

# 5. Generate and Save Confusion Matrix Plot
cm = confusion_matrix(y_test, y_predict, labels=classes)
plt.figure(figsize=(14, 11))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues', 
    xticklabels=classes, 
    yticklabels=classes,
    cbar=True,
    linewidths=0.5
)
plt.title(f'ASL Recognition - Confusion Matrix (Accuracy: {score * 100:.2f}%)', fontsize=15, pad=15)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('Ground Truth Label', fontsize=12)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()

cm_filename = 'confusion_matrix.png'
plt.savefig(cm_filename, dpi=300)
plt.close()
print(f"Confusion matrix saved successfully as '{cm_filename}'.")

# 6. Save the trained AI model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("Trained model saved to 'model.p'. Ready for real-time inference!")
print("=" * 60)