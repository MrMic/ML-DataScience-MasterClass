import marimo

__generated_with = "0.12.9"
app = marimo.App(width="full")


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
def _(X, train_test_split, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=101
    )
    return X_test, X_train, y_test, y_train


@app.cell
def _(mo):
    mo.md(
        """
        ####***TASK:***
        - Dataset features has variety of scales & units => Scale X features
        - Pay attention of what to use for **.fit()** and what to use for **.tranform()**
        """
    )
    return


@app.cell
def _():
    from sklearn.preprocessing import StandardScaler
    return (StandardScaler,)


@app.cell
def _(StandardScaler):
    scaler = StandardScaler()
    return (scaler,)


@app.cell
def _(X_train, scaler):
    # scaler.fir(X_train)
    # scaled_X_train = scaler.transform(X_train)
    scaled_X_train = scaler.fit_transform(X_train)
    return (scaled_X_train,)


@app.cell
def _(X_test, scaler):
    scaled_X_test = scaler.transform(X_test)
    return (scaled_X_test,)


@app.cell
def _(mo):
    mo.md("""
    ###***Test:***
    - Use of Elastic Net model
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
