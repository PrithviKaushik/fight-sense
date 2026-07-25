import marimo

__generated_with = "0.23.14"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exploratory Data Analysis

    The goal of this notebook is to understand the UFC dataset through exploratory data analysis (EDA). We analyze individual feature distributions, investigate relationships between features and fight outcomes, and identify patterns that may be useful for feature engineering and predictive modeling.

    ## Objectives

    - Understand the distribution of important variables.
    - Identify unusual values and potential outliers.
    - Explore relationships between features and fight outcomes.
    - Generate hypotheses for feature engineering.
    """)
    return


@app.cell
def _():
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    return pd, plt, sns


@app.cell
def _(pd):
    ufc_data_path = '../data/raw/ufc-master.csv'
    ufc_data = pd.read_csv(ufc_data_path)
    return (ufc_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Univariate Analysis
    ### Candidate Numerical Variables
    """)
    return


@app.cell
def _(ufc_data):
    ufc_data.describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - R_age
    - B_age

    - R_Reach_cms
    - B_Reach_cms

    - R_Height_cms
    - B_Height_cms

    - R_wins
    - B_wins

    - R_losses
    - B_losses

    - R_odds
    - B_odds
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Age
    """)
    return


@app.cell
def _(pd, ufc_data):
    all_ages = pd.concat([ufc_data['R_age'], ufc_data['B_age']], ignore_index=True)
    return (all_ages,)


@app.cell
def _(all_ages):
    all_ages.describe()
    return


@app.cell
def _(all_ages, sns):
    sns.displot(data = all_ages)
    return


@app.cell
def _(all_ages, plt):
    plt.boxplot(all_ages)
    plt.title("Distribution of Fighter Age")
    plt.ylabel("Age (years)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Reach
    """)
    return


@app.cell
def _(pd, ufc_data):
    all_reach = pd.concat([ufc_data['R_Reach_cms'], ufc_data['B_Reach_cms']], ignore_index=True)
    return (all_reach,)


@app.cell
def _(all_reach):
    all_reach.describe()
    return


@app.cell
def _(all_reach):
    all_reach.hist(bins=100)
    return


@app.cell
def _(all_reach, plt):
    plt.boxplot(all_reach)
    plt.title("Distribution of Fighter Reach")

    plt.ylabel("Reach (cm)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Height
    """)
    return


@app.cell
def _(pd, ufc_data):
    all_height = pd.concat([ufc_data['R_Height_cms'], ufc_data['B_Height_cms']], ignore_index= True)
    return (all_height,)


@app.cell
def _(all_height):
    all_height.describe()
    return


@app.cell
def _(all_height, sns):
    sns.displot(all_height)
    return


@app.cell
def _(all_height, plt):
    plt.boxplot(all_height)
    plt.title("Distribution of Fighter Heights")
    plt.ylabel("Height (cm)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Wins
    """)
    return


@app.cell
def _(ufc_data):
    ufc_data[['R_wins', 'B_wins']]
    return


@app.cell
def _(pd, ufc_data):
    all_wins = pd.concat([ufc_data['R_wins'], ufc_data['B_wins']])
    return (all_wins,)


@app.cell
def _(all_wins):
    all_wins.describe()
    return


@app.cell
def _(all_wins, sns):
    sns.displot(all_wins)
    return


@app.cell
def _(all_wins, plt):
    plt.boxplot(all_wins)
    plt.title("Distribution of Fighter wins")
    plt.ylabel("Number of Wins")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Candidate Categorical Variables
    """)
    return


@app.cell
def _(ufc_data):
    ufc_data.describe(include=['object', 'category', 'str'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - Winner

    - weight_class

    - finish

    - gender

    - title_bout
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### a. Countplot
    """)
    return


@app.cell
def _(sns, ufc_data):
    sns.countplot(ufc_data['weight_class'])
    return


@app.cell
def _(sns, ufc_data):
    sns.countplot(ufc_data['finish'])
    return


@app.cell
def _(ufc_data):
    ufc_data['gender'].value_counts().plot(kind='pie', autopct = '%.2f%%')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Bivariate Analysis

    ### Investigations

    1. Does reach advantage matter?
    2. Do younger fighters win more often?
    3. Do betting favorites actually win?
    4. Does ranking predict the winner?
    5. Does experience matter?
    6. Finish types by weight class
    7. Evolution of the UFC
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Investigation 1 - Reach advantage

    #### Hypothesis
    I believe fighters with a reach advantage are more likely to win because a longer reach allows them to strike from a safer distance.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Create analysis dataset

    Use a boolean mask to filter fights with non-zero reach difference
    """)
    return


@app.cell
def _(ufc_data):
    non_zero_reach_mask = ufc_data['reach_dif'] != 0
    return (non_zero_reach_mask,)


@app.cell
def _(non_zero_reach_mask, ufc_data):
    non_zero_reach_df = ufc_data[non_zero_reach_mask]
    non_zero_reach_df.head()
    return (non_zero_reach_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Filter fights where reach advantage won
    """)
    return


@app.cell
def _(non_zero_reach_df):
    reach_advantage_mask = (((non_zero_reach_df['reach_dif'] > 0) & (non_zero_reach_df['Winner'] == 'Red')) | ((non_zero_reach_df['reach_dif'] < 0) & (non_zero_reach_df['Winner'] == 'Blue')))
    return (reach_advantage_mask,)


@app.cell
def _(reach_advantage_mask):
    reach_advantage_counts = reach_advantage_mask.value_counts()
    return (reach_advantage_counts,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Compute empirical probability
    """)
    return


@app.cell
def _(reach_advantage_mask):
    reach_advantage_mask.mean()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Validation
    """)
    return


@app.cell
def _(non_zero_reach_df, pd, reach_advantage_mask):
    pd.DataFrame({
        "reach_dif": non_zero_reach_df["reach_dif"],
        "Winner": non_zero_reach_df["Winner"],
        "Mask": reach_advantage_mask
    }).sample(15, random_state=42)
    return


@app.cell
def _(non_zero_reach_df):
    non_zero_reach_df[
        ["reach_dif", "Winner"]
    ].head(10)
    return


@app.cell
def _(non_zero_reach_df):
    red_reach_wins = (
        (non_zero_reach_df["reach_dif"] > 0)
        & (non_zero_reach_df["Winner"] == "Red")
    )

    blue_reach_wins = (
        (non_zero_reach_df["reach_dif"] < 0)
        & (non_zero_reach_df["Winner"] == "Blue")
    )

    (red_reach_wins.sum() + blue_reach_wins.sum()) / len(non_zero_reach_df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Plot
    """)
    return


@app.cell
def _(plt, reach_advantage_counts):
    reach_advantage_counts.plot(kind='bar', color='purple')

    plt.tight_layout()
    plt.xlabel('Reached-advantaged Fighter Won')
    plt.ylabel('Number of Fights')
    plt.title('Fight Outcomes by Reach Advantage')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Findings

    - Excluded fights where neither fighter had a reach advantage (`reach_dif == 0`).
    - Among the remaining fights, the fighter with the reach advantage won **48.099%** of the time.
    - The observed win rate (48.1%) is below 50%, suggesting that reach advantage alone does not provide an observable advantage in this dataset.
    - The initial hypothesis is not supported by this analysis.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Conclusion

    Based on this dataset, reach advantage alone does not appear to increase the probability of winning a UFC fight.

    However, this analysis considers only reach difference in isolation. Other variables such as age, experience, betting odds, rankings, and weight class may interact with reach and should be investigated before drawing stronger conclusions.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Investigation 2 - Age factor

    #### Hypothesis
    I believe younger fighters are more likely to win because they generally have better athleticism and recovery.
    """)
    return


@app.cell
def _(ufc_data):
    unequal_age_mask = ufc_data['R_age'] != ufc_data['B_age']
    return (unequal_age_mask,)


@app.cell
def _(ufc_data, unequal_age_mask):
    unequal_age_df = ufc_data[unequal_age_mask]
    unequal_age_df.head()
    return (unequal_age_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Filter fights where younger fighter won
    """)
    return


@app.cell
def _(unequal_age_df):
    age_advantage_mask = ((unequal_age_df['R_age'] < unequal_age_df['B_age']) & (unequal_age_df['Winner'] == 'Red') | ((unequal_age_df['B_age'] < unequal_age_df['R_age']) & (unequal_age_df['Winner'] == 'Blue')))
    return (age_advantage_mask,)


@app.cell
def _(age_advantage_mask):
    age_advantage_counts = age_advantage_mask.value_counts()
    print(age_advantage_counts)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Compute Empirical Probability
    """)
    return


@app.cell
def _(age_advantage_mask):
    younger_fighter_win_rate = age_advantage_mask.mean()
    print(younger_fighter_win_rate)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Findings

    - After excluding fights where both fighters were the same age, the younger fighter won approximately **57.3%** of the fights.
    - This suggests that younger fighters have a higher observed win rate than older fighters.
    - The effect appears stronger than the reach advantage investigated previously.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Conclusion

    The results provide evidence that age is associated with fight outcomes. In this dataset, the younger fighter won more often than the older fighter.

    However, age alone is unlikely to determine the outcome of a fight. Other factors such as experience, rankings, betting odds, and fighting style may also influence the result and should be investigated.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Investigation 3 - Betting odds
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Inspecting data
    """)
    return


@app.cell
def _(ufc_data):
    ufc_data[['R_odds', 'B_odds']].head(10)
    return


@app.cell
def _(ufc_data):
    ufc_data[['R_odds', 'B_odds']].describe()
    return


@app.cell
def _(ufc_data):
    ufc_data['R_odds'].value_counts().head(10)
    return


@app.cell
def _(ufc_data):
    ufc_data[['R_fighter','B_fighter','R_odds', 'B_odds', 'Winner']].sample(10, random_state=42)
    return


@app.cell
def _(ufc_data):
    ufc_data[ufc_data["R_odds"] == ufc_data["B_odds"]]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Hypothesis

    I believe the fighter with lower betting odds, i.e., the favoured fighter is more likely to win.
    """)
    return


@app.cell
def _(ufc_data):
    non_equal_odds_df = ufc_data[ufc_data["R_odds"] != ufc_data["B_odds"]]
    return (non_equal_odds_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Filtering fights where favoured fighter won
    """)
    return


@app.cell
def _(non_equal_odds_df):
    favourite_fighter_mask = (((non_equal_odds_df['R_odds'] < non_equal_odds_df['B_odds']) & (non_equal_odds_df['Winner'] == 'Red')) | (non_equal_odds_df['R_odds'] > non_equal_odds_df['B_odds']) & (non_equal_odds_df['Winner'] == 'Blue'))
    return (favourite_fighter_mask,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Calculating Empirical Probability
    """)
    return


@app.cell
def _(favourite_fighter_mask):
    favourite_fighter_win_rate = favourite_fighter_mask.mean()
    print (favourite_fighter_win_rate)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Plot
    """)
    return


@app.cell
def _(favourite_fighter_mask):
    favourite_fighter_counts = favourite_fighter_mask.value_counts()
    return (favourite_fighter_counts,)


@app.cell
def _(favourite_fighter_counts, plt):
    favourite_fighter_counts.plot(kind="bar", color = 'green')
    favourite_fighter_counts.index = ["Underdog Won", "Favourite Won"]

    plt.xlabel("Favourite Won")
    plt.ylabel("Number of Fights")
    plt.title("Fight Outcomes for Betting Favorites")

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Findings
    - After excluding fights with equal betting odds, the betting favorite won approximately **64.1%** of the fights.
    - This is the strongest predictor investigated so far.
    - Although betting favorites win more often than underdogs, upsets still occur in roughly one-third of fights.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Conclusion
    Betting odds provide a stronger signal than reach or age alone. This suggests that betting markets successfully aggregate information about fighters before a bout. However, betting odds are not perfect predictors, as underdogs still win a substantial proportion of fights.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Investigation 4 - Rank Advantage
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Hypothesis
    I believe higher ranked fighters are more likely to win because of superior performance.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Inspecting the data
    """)
    return


@app.cell
def _(ufc_data):
    ufc_data[
        [
            'better_rank', 'R_match_weightclass_rank', 'B_match_weightclass_rank'
        ]
    ].head()
    return


@app.cell
def _(ufc_data):
    ufc_data[
        [
            "R_match_weightclass_rank",
            "B_match_weightclass_rank",
            "better_rank"
        ]
    ].isna().sum()
    return


@app.cell
def _(ufc_data):
    ufc_data["better_rank"].value_counts(dropna=False)
    return


@app.cell
def _(ufc_data):
    ufc_data["better_rank"].unique()
    return


@app.cell
def _(ufc_data):
    ufc_data.loc[
        ufc_data["better_rank"] == "Red",
        [
            "R_match_weightclass_rank",
            "B_match_weightclass_rank",
            "better_rank"
        ]
    ].head(10)
    return


@app.cell
def _(ufc_data):
    ufc_data.loc[
        ufc_data["better_rank"] == "Blue",
        [
            "R_match_weightclass_rank",
            "B_match_weightclass_rank",
            "better_rank"
        ]
    ].head(10)
    return


@app.cell
def _(ufc_data):
    ufc_data.loc[
        ufc_data["better_rank"] == "neither",
        [
            "R_match_weightclass_rank",
            "B_match_weightclass_rank",
            "better_rank"
        ]
    ].head(10)
    return


if __name__ == "__main__":
    app.run()
