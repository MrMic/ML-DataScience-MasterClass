import marimo

__generated_with = "0.13.9"
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
    numeric_df = df.drop("species", axis=1)
    sns.heatmap(numeric_df.corr(), annot=True)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
