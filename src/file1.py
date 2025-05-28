import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_mlflow():
    print("Tracking URI:", mlflow.get_tracking_uri())
    mlflow.set_tracking_uri("http://127.0.0.1:5001")
    print("Tracking URI:", mlflow.get_tracking_uri())

    mlflow.set_experiment('Experiment1')

    wine = load_wine()
    X = wine.data
    y = wine.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

    max_depth = 10
    n_estimators = 10


    # Define your custom artifact directory
    script_dir = os.path.dirname(__file__)
    artifact_dir = os.path.join(script_dir, "mlartifacts")
    os.makedirs(artifact_dir, exist_ok=True)  # create if doesn't exist

    try:
        with mlflow.start_run():
            rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
            rf.fit(X_train, y_train)

            y_pred = rf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            print("Accuracy:", accuracy)

            mlflow.log_metric('accuracy', accuracy)
            mlflow.log_param('max_depth', max_depth)
            mlflow.log_param('n_estimators', n_estimators)

            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6,6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
            plt.title("Confusion Matrix")
            plt.savefig("Confusion-matrix.png")

           # Save and log plot to `mlartifacts`
            cm_path = os.path.join(artifact_dir, "confusion_matrix.png")
            plt.savefig(cm_path)
            plt.close()

            mlflow.log_artifact(cm_path)
            # tags
            mlflow.set_tags({"Author": 'Vikash', "Project": "Wine Classification"})

            # Log the model
            mlflow.sklearn.log_model(rf, "Random-Forest-Model")

    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    run_mlflow()
