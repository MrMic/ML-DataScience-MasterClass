import marimo

__generated_with = "0.12.10"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        mo.md(\"""
        ## CROSS VALIDATION PROJECT **-= EXERCISE =-**
        \""")
        """
    )
    return


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
    mo.md(
        """
        ###***Task:***
        - Use of Elastic Net model
        """
    )
    return


@app.cell
def _():
    from sklearn.linear_model import ElasticNet
    return (ElasticNet,)


@app.cell
def _(ElasticNet):
    base_elastic_model = ElasticNet(max_iter=500000)
    return (base_elastic_model,)


@app.cell
def _():
    param_grid = {"alpha": [0.1, 1, 5, 100, 100], "l1_ratio": [0.1, 0.7, 0.99, 1]}
    return (param_grid,)


@app.cell
def _(mo):
    mo.md(
        """
        ###***Task:***
        - Use Scikit-lean to create GridSearchCV & run a grid search for the best 
        parameters for your model based on your scaled training data.
        """
    )
    return


@app.cell
def _():
    from sklearn.model_selection import GridSearchCV
    return (GridSearchCV,)


@app.cell
def _(GridSearchCV, base_elastic_model, param_grid):
    grid_model = GridSearchCV(
        estimator=base_elastic_model,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=5,
        verbose=1,
    )
    return (grid_model,)


@app.cell
def _(grid_model, scaled_X_train, y_train):
    grid_model.fit(X=scaled_X_train, y=y_train)
    return


@app.cell
def _(mo):
    mo.md(
        """
        ###***Task:***
        - Display the best combination of parameters for your model
        """
    )
    return


@app.cell
def _(grid_model):
    grid_model.best_params_
    return


@app.cell
def _(mo):
    mo.md(
        """
        ###***Task:***
        - Evaluate your model performance on the unseen 10% scaled test set.
        """
    )
    return


@app.cell
def _(grid_model, scaled_X_test):
    y_pred = grid_model.predict(scaled_X_test)
    return (y_pred,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
