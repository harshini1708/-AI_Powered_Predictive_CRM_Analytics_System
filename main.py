import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
data = pd.read_csv('customer_data.csv')

# Preprocessing
X = data.drop('retention', axis=1)
y = data['retention']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train RandomForest as a baseline model
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print(f"RandomForest Accuracy: {accuracy_score(y_test, rf_pred)}")

# Confusion matrix
rf_cm = confusion_matrix(y_test, rf_pred)
sns.heatmap(rf_cm, annot=True, fmt="d")
plt.title('Confusion Matrix for RandomForest')
plt.show()

# Build a Keras deep learning model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate model
dl_pred = (model.predict(X_test) > 0.5).astype("int32")
print(f"Deep Learning Model Accuracy: {accuracy_score(y_test, dl_pred)}")

# Confusion matrix for deep learning
dl_cm = confusion_matrix(y_test, dl_pred)
sns.heatmap(dl_cm, annot=True, fmt="d")
plt.title('Confusion Matrix for Deep Learning Model')
plt.show()

# Save model
model.save('predictive_crm_model.h5')
