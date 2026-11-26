"""Train a simple Iris flower classifier and make one sample prediction."""

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main() -> None:
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42,
        stratify=iris.target,
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Test accuracy: {accuracy:.1%}")

    # Measurements are in centimeters: sepal length/width, petal length/width.
    sample = [[5.1, 3.5, 1.4, 0.2]]
    predicted_class = model.predict(sample)[0]
    print(f"Sample flower prediction: {iris.target_names[predicted_class]}")


if __name__ == "__main__":
    main()