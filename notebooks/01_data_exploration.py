import marimo

__generated_with = "0.23.15"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Dataset Exploration

    ## Objective

    Understand the UFC dataset before performing any analysis.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Import Libraries
    """)
    return


@app.cell
def _():
    import pandas as pd

    return (pd,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load Dataset
    """)
    return


@app.cell
def _(pd):
    # save filepath to variable for easier access
    ufc_file_path = '../data/raw/ufc-master.csv'

    # read the data and store data in DataFrame titled ufc_data
    ufc_data = pd.read_csv(ufc_file_path)
    return (ufc_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dataset Overview
    """)
    return


@app.cell
def _(ufc_data):
    # print summary of data
    ufc_data.describe()
    return


@app.cell
def _(ufc_data):
    ufc_data.info()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What does one row represent?
    """)
    return


@app.cell
def _(ufc_data):
    ufc_data.columns.to_list()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Findings

    - Calling `dropna()` removes every row.
    - Missing values are largely structural.
    - Women's ranking columns are mostly empty because most fights aren't in those divisions.
    """)
    return


@app.cell
def _(ufc_data):
    ufc_data.shape
    return


@app.cell
def _(ufc_data):
    ufc_data.isna().sum().sort_values(ascending=False)
    return


@app.cell
def _(ufc_data):
    print(ufc_data[["Winner"]])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Find every unique weight class.
    """)
    return


@app.cell
def _(ufc_data):
    ufc_data["weight_class"].unique()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Which division has most fights?
    """)
    return


@app.cell
def _(ufc_data):
    ufc_data['weight_class'].value_counts()
    return


@app.cell
def _(ufc_data):
    ufc_data['weight_class'].value_counts(normalize=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Every time we investigate a categorical column, ask these three questions:

    - What categories exist?
        unique()
    - How many of each are there?
        value_counts()
    - What proportion does each category represent?
        value_counts(normalize=True)

    You'll use this pattern dozens of times in this project.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Target Variable
    """)
    return


@app.cell
def _(ufc_data):
    y = ufc_data['Winner']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Is the dataset balanced about the Winning variable?
    """)
    return


@app.cell
def _(ufc_data):
    ufc_data['Winner'].value_counts(normalize=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Findings

    Winner distribution:

    - Red: 57.75%
    - Blue: 42.13%
    - Draw: very rare
    - No Contest: very rare

    A baseline model predicting "Red" would achieve about 58% accuracy.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Datatypes
    """)
    return


@app.cell
def _(ufc_data):
    ufc_data.dtypes
    return


@app.cell
def _(ufc_data):
    ufc_data.info()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checking for duplicates
    """)
    return


@app.cell
def _(ufc_data):
    ufc_data.duplicated().sum()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    No duplicates present, would've used pd.DataFrame.drop_duplicates if needed.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - Each row represents one UFC fight between a Red and Blue fighter.
    - The dataset contains 7,177 fights and 118 features.
    - Missing values are often structural (e.g., ranking columns for irrelevant weight classes), so `dropna()` is not an appropriate first cleaning step.
    - The target variable (`Winner`) is reasonably balanced (~58% Red, ~42% Blue), with very few Draws and No Contests.
    - The dataset contains no duplicate rows.
    - Several columns (e.g., `finish_round`, `finish_details`, `total_fight_time_secs`) would cause data leakage if used as features.

    ## Next Question

    Does having a reach advantage increase the probability of winning?
    """)
    return


if __name__ == "__main__":
    app.run()
