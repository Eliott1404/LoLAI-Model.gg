import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn

# Set the tracking URI to your local filesystem (default)
mlflow.set_tracking_uri("http://localhost:5000")

# Set the experiment name
mlflow.set_experiment("League of Legends Win Prediction")

# Load the dataset
file_path = 'data/transformed_games_final_one_hot.csv'
data = pd.read_csv(file_path)

# Separate the features and the target variable
X = data.drop(columns=['winner'])
y = data['winner']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start an MLflow run
with mlflow.start_run():
    # Train a logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Log model
    mlflow.sklearn.log_model(model, "logistic_regression_model_v1.0")

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    for class_label, metrics in report.items():
        if class_label not in ('accuracy', 'macro avg', 'weighted avg'):
            mlflow.log_metric(f'precision_{class_label}', metrics['precision'])
            mlflow.log_metric(f'recall_{class_label}', metrics['recall'])
            mlflow.log_metric(f'f1-score_{class_label}', metrics['f1-score'])

    # Print results
    print(f'Accuracy: {accuracy}')
    print(f'Classification Report:\n{classification_report(y_test, y_pred)}')
