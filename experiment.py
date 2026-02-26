"""
Experimental comparison between:
- Custom Decision Tree
- Custom Random Forest
- sklearn Decision Tree
- sklearn Random Forest
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier as SkDecisionTree
from sklearn.ensemble import RandomForestClassifier as SkRandomForest

from decision_tree import DecisionTreeClassifier
from random_forest import RandomForestClassifier
from utils import measure_time


def main():

    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Custom DT": DecisionTreeClassifier(max_depth=10),
        "Custom RF": RandomForestClassifier(n_estimators=20, max_depth=10,
                                            max_features=int(np.sqrt(X.shape[1]))),
        "Sklearn DT": SkDecisionTree(max_depth=10),
        "Sklearn RF": SkRandomForest(n_estimators=20, max_depth=10)
    }

    train_acc = []
    test_acc = []
    train_time = []

    for name, model in models.items():
        _, t = measure_time(model.fit, X_train, y_train)
        train_time.append(t)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_acc.append(accuracy_score(y_train, train_pred))
        test_acc.append(accuracy_score(y_test, test_pred))

        print(f"{name}")
        print(f"Train Accuracy: {train_acc[-1]:.4f}")
        print(f"Test Accuracy : {test_acc[-1]:.4f}")
        print(f"Training Time : {t:.4f} sec\n")

    # Accuracy Plot 
    labels = list(models.keys())
    x = np.arange(len(labels))

    plt.figure(figsize=(10, 5))
    plt.bar(x - 0.2, train_acc, 0.4, label="Train Accuracy")
    plt.bar(x + 0.2, test_acc, 0.4, label="Test Accuracy")
    plt.xticks(x, labels)
    plt.legend()
    plt.title("Accuracy Comparison")
    plt.show()

    # Training Time Plot 
    plt.figure(figsize=(8, 5))
    plt.bar(labels, train_time)
    plt.title("Training Time Comparison")
    plt.ylabel("Seconds")
    plt.show()


if __name__ == "__main__":
    main()
