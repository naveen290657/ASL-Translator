import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load extracted landmarks
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split into 80% training data, 20% testing data
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train the Random Forest Model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Test accuracy
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print(f'Model Training Complete! Accuracy: {score * 100:.2f}%')

# Save the trained AI model
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()