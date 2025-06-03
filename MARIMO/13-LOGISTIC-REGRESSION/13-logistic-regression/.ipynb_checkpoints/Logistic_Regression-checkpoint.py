import marimo

__generated_with = "0.13.8"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    return pd, plt, sns


@app.cell
def _(pd):
    df = pd.read_csv("./hearing_test.csv")
    return (df,)


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(df):
    df.describe()
    return


@app.cell
def _(df):
    df["test_result"].value_counts()
    return


@app.cell
def _(df, sns):
    sns.countplot(data=df, x=df["test_result"])
    return


@app.cell
def _(df, mo, sns):
    mo.mpl.interactive(sns.countplot(data=df, x=df["test_result"]))
    return


@app.cell
def _(df, plt, sns):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="test_result", y="physical_score", data=df)
    return


@app.cell
def _(df, plt, sns):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x="age", y="physical_score", data=df, hue="test_result", alpha=0.5
    )
    return


@app.cell
def _(df, sns):
    sns.pairplot(data=df, hue="test_result")
    return


@app.cell
def _(df, sns):
    sns.heatmap(df.corr(), annot=True)
    return


@app.cell
def _(df, sns):
    sns.scatterplot(x="physical_score", y="test_result", data=df)
    return


@app.cell
def _(df, plt):
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        df["age"], df["physical_score"], df["test_result"], c=df["test_result"]
    )
    return (ax,)


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(df):
    X = df.drop("test_result", axis=1)
    return (X,)


@app.cell
def _(df):
    y = df["test_result"]
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
        X, y, test_size=0.1, random_state=101
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
    scaled_X_test = scaler.transform(X_test)
    return (scaled_X_test,)


@app.cell
def _():
    from sklearn.linear_model import LogisticRegression
    return (LogisticRegression,)


@app.cell
def _(LogisticRegression):
    log_model = LogisticRegression()
    return (log_model,)


@app.cell
def _(log_model, scaled_X_train, y_train):
    log_model.fit(scaled_X_train, y_train)
    return


@app.cell
def _(log_model):
    log_model.coef_
    return


@app.cell
def _(log_model, scaled_X_test):
    y_pred = log_model.predict(scaled_X_test)
    return (y_pred,)


@app.cell
def _(y_pred):
    y_pred
    return


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _():
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        classification_report,
    )
    return accuracy_score, classification_report, confusion_matrix


@app.cell
def _(confusion_matrix, y_pred, y_test):
    confusion_matrix(y_test, y_pred)
    return


@app.cell
def _(accuracy_score, y_pred, y_test):
    accuracy_score(y_test, y_pred)
    return


@app.cell
def _():
    from sklearn.metrics import ConfusionMatrixDisplay
    return (ConfusionMatrixDisplay,)


@app.cell
def _(ConfusionMatrixDisplay, confusion_matrix, plt, y_pred, y_test):
    # Assuming y_true and y_pred are your actual and predicted labels
    # cm = confusion_matrix(y_test, y_pred,normalize="all")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
    )
    disp.plot()
    plt.show()
    return


@app.cell
def _(y_test):
    len(y_test)
    return


@app.cell
def _(classification_report, y_pred, y_test):
    print(classification_report(y_pred=y_test, y_true=y_pred))
    return


@app.cell
def _():
    from sklearn.metrics import precision_score, recall_score
    return precision_score, recall_score


@app.cell
def _(precision_score, y_pred, y_test):
    precision_score(y_test, y_pred=y_pred)
    return


@app.cell
def _(recall_score, y_pred, y_test):
    recall_score(y_true=y_test, y_pred=y_pred)
    return


@app.cell
def _():
    from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
    return PrecisionRecallDisplay, RocCurveDisplay


@app.cell
def _(RocCurveDisplay, ax, log_model, plt, scaled_X_test, y_test):
    fig2, ax2 = plt.subplots(figsize=(6, 4))

    # First curve
    display1 = RocCurveDisplay.from_estimator(
        log_model, scaled_X_test, y_test, name="Model 'log_model'", ax=ax2
    )

    ax.set_title("ROC Curves")
    plt.show()
    return


@app.cell
def _(PrecisionRecallDisplay, ax, log_model, plt, scaled_X_test, y_test):
    fig3, ax3 = plt.subplots(figsize=(12, 6))

    # First curve
    display2 = PrecisionRecallDisplay.from_estimator(
        log_model, scaled_X_test, y_test, name="Model 'log_model'", ax=ax3
    )

    ax.set_title("ROC Curves")
    plt.show()
    return


@app.cell
def _(log_model, scaled_X_test):
    log_model.predict_proba(scaled_X_test)[0]
    return


@app.cell
def _(y_test):
    y_test[0]
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
