import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import argparse

def load_embeddings_and_labels(embeddings_path):
    with open(embeddings_path, 'rb') as f:
        data = pickle.load(f)
    return data

def train_xgboost(X_train, y_train, X_test, y_test):
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    X_train_2d = np.vstack(X_train)
    X_test_2d = np.vstack(X_test)
    xgb_model.fit(X_train_2d, y_train)

    # Evaluate the model
    y_pred = xgb_model.predict(X_test_2d)
    report_metrics("XGBoost", y_test, y_pred)

    return xgb_model

def train_catboost(X_train, y_train, X_test, y_test):
    catboost_model = CatBoostClassifier(logging_level='Silent')
    X_train_2d = np.vstack(X_train)
    X_test_2d = np.vstack(X_test)
    catboost_model.fit(X_train_2d, y_train)

    # Evaluate the model
    y_pred = catboost_model.predict(X_test_2d)
    report_metrics("CatBoost", y_test, y_pred)

    return catboost_model

def report_metrics(model_name, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    print(f"{model_name} Model:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("\n")

def main():
    parser = argparse.ArgumentParser(description="Building model from embeddings")
    parser.add_argument("-algorithm", choices=["xgb", "cat"], required=True,
                        help="Select algorithm model")
    parser.add_argument("-embeddingfile", required=True,
                        help="Relative path to the embedding file")

    args = parser.parse_args()
    # Specify the path to the file containing embeddings and labels
    embeddings_file = os.path.join("embeddings", args.embeddingfile)

    # Load embeddings and labels
    all_data = load_embeddings_and_labels(embeddings_file)
    embeddings, labels = zip(*all_data)

    # Split the data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

    if args.algorithm == "xgb":
        model = train_xgboost(X_train, y_train, X_test, y_test)
    elif args.algorithm == "cat":
        model = train_catboost(X_train, y_train, X_test, y_test)
    else:
        raise ValueError("Invalid algorithm. Available options: 'xgb', 'cat'")

    output_file = f"model/model_{args.algorithm}.pkl"

    # Save the trained models to disk (serialization)
    save_model(model, output_file)


def save_model(model, output_path):
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    main()
