"""Train the Iris classifier, then predict a species from terminal input."""

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

    print("\nEnter the flower measurements in centimeters.")
    print("Order: sepal length, sepal width, petal length, petal width")
    print("Example: 5.1, 3.5, 1.4, 0.2")

    while True:
        raw_measurements = input("Measurements: ").strip()
        try:
            measurements = [float(value.strip()) for value in raw_measurements.split(",")]
            if len(measurements) != 4 or any(value <= 0 for value in measurements):
                raise ValueError
        except ValueError:
            print("Please enter exactly four positive numbers, separated by commas.")
            continue
        break

    predicted_class = model.predict([measurements])[0]
    print(f"Predicted species: {iris.target_names[predicted_class].title()}")


if __name__ == "__main__":
    main()
