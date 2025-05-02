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
def _():
    return


if __name__ == "__main__":
    app.run()
