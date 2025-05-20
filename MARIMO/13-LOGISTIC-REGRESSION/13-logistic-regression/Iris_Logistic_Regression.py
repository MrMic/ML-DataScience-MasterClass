import marimo

__generated_with = "0.13.10"
app = marimo.App(width="full")


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    return np, pd, plt, sns


@app.cell
def _(pd):
    df = pd.read_csv("iris.csv")
    return (df,)


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(df):
    df.info()
    return


@app.cell
def _(df):
    df.describe()
    return


@app.cell
def _(df):
    df["species"].value_counts()
    return


@app.cell
def _(df, sns):
    sns.countplot(data=df, x="species")
    return


@app.cell
def _(df, sns):
    sns.scatterplot(data=df, x="petal_length", y="petal_width", hue="species")
    return


@app.cell
def _(df, sns):
    sns.pairplot(data=df, hue="species")
    return


@app.cell
def _(df, sns):
    # Calculate correlation only on numeric columns
    X = df.drop("species", axis=1)
    sns.heatmap(X.corr(), annot=True)
    return (X,)


@app.cell
def _(df):
    y = df["species"]
    return (y,)


@app.cell
def _():
    from sklearn.model_selection import train_test_split
    return (train_test_split,)


@app.cell
def _():
    from sklearn.preprocessing import StandardScaler
    return (StandardScaler,)


@app.cell
def _(X, train_test_split, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=101
    )
    return X_test, X_train, y_test, y_train


@app.cell
def _(StandardScaler):
    scaler = StandardScaler()
    return (scaler,)


@app.cell
def _(X_train, scaler):
    scaled_X_train = scaler.fit_transform(X_train)
    return (scaled_X_train,)


@app.cell
def _(X_test, scaler):
    scaled_X_test = scaler.fit_transform(X_test)
    return (scaled_X_test,)


@app.cell
def _():
    from sklearn.linear_model import LogisticRegression
    return (LogisticRegression,)


@app.cell
def _():
    from sklearn.model_selection import GridSearchCV
    return (GridSearchCV,)


@app.cell
def _(LogisticRegression):
    log_model = LogisticRegression(solver="saga", multi_class="ovr", max_iter=500)
    return (log_model,)


@app.cell
def _(np):
    penalty = ["l1", "l2", "elasticnet"]
    l1_ratio = np.linspace(0, 1, 20)
    C = np.logspace(0, 10, 20)

    param_grid = {"penalty": penalty, "l1_ratio": l1_ratio, "C": C}
    return (param_grid,)


@app.cell
def _(GridSearchCV, log_model, param_grid):
    grid_model = GridSearchCV(log_model, param_grid=param_grid)
    return (grid_model,)


@app.cell
def _(grid_model, scaled_X_train, y_train):
    grid_model.fit(scaled_X_train, y_train)
    return


@app.cell
def _():
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        classification_report,
        ConfusionMatrixDisplay,
    )
    return (
        ConfusionMatrixDisplay,
        accuracy_score,
        classification_report,
        confusion_matrix,
    )


@app.cell
def _(grid_model):
    grid_model.best_params_
    return


@app.cell
def _(grid_model, scaled_X_test):
    y_pred = grid_model.predict(scaled_X_test)
    return (y_pred,)


@app.cell
def _(y_pred):
    y_pred
    return


@app.cell
def _(accuracy_score, y_pred, y_test):
    accuracy_score(y_test, y_pred)
    return


@app.cell
def _(confusion_matrix, y_pred, y_test):
    confusion_matrix(y_test, y_pred)
    return


@app.cell
def _(ConfusionMatrixDisplay, confusion_matrix, plt, y_pred, y_test):
    # Method 1: Current recommended approach with ConfusionMatrixDisplay
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()
    return


@app.cell
def _(ConfusionMatrixDisplay, plt, y_pred, y_test):
    # Method 2: Alternative approach directly from predictions
    disp2 = ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, cmap=plt.cm.Blues
    )
    plt.title("Confusion Matrix from Predictions")
    plt.show()
    return


@app.cell
def _(ConfusionMatrixDisplay, grid_model, plt, scaled_X_test, y_test):
    # Method 3: Alternative approach directly from estimator
    disp3 = ConfusionMatrixDisplay.from_estimator(
        grid_model, scaled_X_test, y_test, cmap=plt.cm.Blues
    )
    plt.title("Confusion Matrix from Estimator")
    plt.show()
    return


@app.cell
def _(classification_report, y_pred, y_test):
    print(classification_report(y_test, y_pred))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
