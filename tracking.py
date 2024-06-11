import mlflow

# Set the tracking URI to your local filesystem (default)
mlflow.set_tracking_uri("http://localhost:5000")

# Set the experiment name
mlflow.set_experiment("League of Legends Win Prediction")

# Run the training script
if __name__ == "__main__":
    mlflow.run("scripts/train.py")
