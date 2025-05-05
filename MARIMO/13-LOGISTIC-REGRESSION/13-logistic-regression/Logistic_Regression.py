import marimo

__generated_with = "0.13.4"
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
    return


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
    return X_test, X_train, y_train


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
    y_pred = log_model.predict_proba(scaled_X_test)
    return (y_pred,)


@app.cell
def _(y_pred):
    y_pred
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
