# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('data - data.csv')

# Select features
selected_features = ['id','diagnosis','radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean',
                    'compactness_mean','concavity_mean','concave points_mean','symmetry_mean',
                    'fractal_dimension_mean','radius_se','texture_se','perimeter_se','area_se','smoothness_se',
                    'compactness_se','concavity_se','concave points_se','symmetry_se','fractal_dimension_se',
                    'radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst',
                    'compactness_worst','concavity_worst','concave points_worst','symmetry_worst',
                    'fractal_dimension_worst']

df_selected = df[selected_features]

# Preprocessing
# Encode the diagnosis column (M = 1, B = 0)
label_encoder = LabelEncoder()
df_selected['diagnosis'] = label_encoder.fit_transform(df_selected['diagnosis'])

# Separate features and target variable
X = df_selected.drop(['id', 'diagnosis'], axis=1)  # Features
y = df_selected['diagnosis']  # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the classifier (Random Forest Classifier)
clf = RandomForestClassifier(random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Classification report
print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))

# Counting occurrences of Benign and Malignant in test set
count_benign = y_test.value_counts()[0]
count_malignant = y_test.value_counts()[1]

# Plot bar graph with reduced width
plt.figure(figsize=(6, 5))  # Adjust figure size if needed
plt.bar(['Benign', 'Malignant'], [count_benign, count_malignant], color=['blue', 'green'], width=0.5)
plt.title('Count of Benign and Malignant Diagnoses')
plt.ylabel('Count')
plt.show()

# Plotting confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = [0, 1]
plt.xticks(tick_marks, ['Benign', 'Malignant'])
plt.yticks(tick_marks, ['Benign', 'Malignant'])

thresh = cm.max() / 2.
for i, j in ((i, j) for i in range(cm.shape[0]) for j in range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
             color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()
