"""Small Flask web app for the Iris classifier, ready for Render."""

from flask import Flask, render_template, request
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier


app = Flask(__name__)

# Train once when the web service starts, then reuse the model for requests.
iris = load_iris()
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(iris.data, iris.target)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None
    values = ["5.1", "3.5", "1.4", "0.2"]

    if request.method == "POST":
        values = [request.form.get(name, "").strip() for name in (
            "sepal_length", "sepal_width", "petal_length", "petal_width"
        )]
        try:
            measurements = [float(value) for value in values]
            if any(value <= 0 for value in measurements):
                raise ValueError
            class_id = model.predict([measurements])[0]
            prediction = str(iris.target_names[class_id]).title()
        except ValueError:
            error = "Enter a positive number for each measurement."

    return render_template(
        "index.html", prediction=prediction, error=error, values=values
    )


if __name__ == "__main__":
    app.run(debug=True)
