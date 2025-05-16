import marimo

__generated_with = "0.13.10"
app = marimo.App(width="full")


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    return pd, sns


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
    return X_test, X_train


@app.cell
def _(StandardScaler):
    scaler = StandardScaler()
    return (scaler,)


@app.cell
def _(X_train, scaler):
    scaled_X_train = scaler.fit_transform(X_train)
    return


@app.cell
def _(X_test, scaler):
    scaled_X_test = scaler.fit_transform(X_test)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
