import marimo

__generated_with = "0.12.8"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""## CROSS VALIDATION PROJECT **-= EXERCISE =-**""")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    return np, pd, plt, sns


@app.cell
def _(pd):
    df = pd.read_csv("AMES_Final_DF.csv")
    return (df,)


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(df):
    df.info()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""**TASK:** Label to predict is the SalesPrice column. Separate Data into X features and y labels.""")
    return


@app.cell
def _(df):
    X = df.drop("SalePrice", axis=1)
    return (X,)


@app.cell
def _(df):
    y = df["SalePrice"]
    return (y,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ###***TASK:***
        - Use Scikit-learn to split up X and y into trainig set andtest set.
        - Test proportion to **10%** because we will be using **Grid Search strategy**.
        - Data split: **_random_state = 101_**
        """
    )
    return


@app.cell
def _():
    from sklearn.model_selection import train_test_split
    return (train_test_split,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
