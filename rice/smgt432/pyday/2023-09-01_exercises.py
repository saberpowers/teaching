

# Step 1: Read the data into Python ----

import pandas as pd

data_raw = pd.read_csv('super_league_2022.csv')     # update file path to where you saved the data

'''
## EXERCISE 1 ##
Create a new pandas dataframe called `data` from `data_raw` with two new columns: `home_goals` and
`away_goals`, by extracting the number of goals for each side from the `score` column.
'''
data = data_raw


# Step 2: Train a Poisson Bradley-Terry model ----

import statsmodels.api as sm
import statsmodels.formula.api as smf

# Wrangle data for the Bradley-Terry model. We need two rows for every game, one for goals scored
# by the home team and one for goals scored by the away team.
goal_model_data = pd.concat(
    objs=[
        data[['home_team','away_team','home_goals']].assign(
            home=1
        ).rename(
            columns={'home_team':'offense', 'away_team':'defense', 'home_goals':'goals'}
        ),
        data[['away_team','home_team','away_goals']].assign(
            home=0
        ).rename(
            columns={'away_team': 'offense', 'home_team':'defense', 'away_goals':'goals'}
        )
    ]
)

'''
## EXERCISE 2 ##
Fit a Poisson Bradley-Terry model to predict number of goals scored using home, offense and defense
'''
poisson_model = None


# Step 3: Produce predictions ----

import numpy as np
from itertools import product
from scipy.stats import skellam

all_teams = np.unique(data['home_team'])

# Create a dataframe with all possible combinations of home (0/1), offense and defense
pred_data = pd.DataFrame(
    product([1, 0], all_teams, all_teams),
    columns=['home', 'offense', 'defense']
).query(
    'offense != defense'    # remove rows where the same team is the offense and the defense
)

pred_raw = pred_data.assign(
    pred_goals=poisson_model.predict(exog=pred_data)
)

pred_home = pred_raw.query(
    'home == 1'
).rename(
    columns={'offense':'home_team', 'defense':'away_team', 'pred_goals':'pred_goals_home'}
).loc[
    :, ['home_team', 'away_team', 'pred_goals_home']
]

pred_away = pred_raw.query(
    'home == 0'
).rename(
    columns={'defense':'home_team', 'offense':'away_team', 'pred_goals':'pred_goals_away'}
).loc[
    :, ['home_team', 'away_team', 'pred_goals_away']
]

# Put home and away goal distributions together and calculate the probability of each outcome
pred = pd.merge(pred_home, pred_away, on=['home_team', 'away_team']).assign(
    '''
    ## EXERCISE 3 ##
    Fill in the probability of the home team winning and the probability of the away team winning.
    Check that prob_home_win + prob_away_win + prob_draw = 1!
    '''
    prob_home_win=0,
    prob_away_win=0,
    prob_draw=lambda x: [
        # Calculate the probability that the Skellam-distributed difference is exactly zero
        skellam.pmf(0, x['pred_goals_home'][i], x['pred_goals_away'][i]) for i in range(0, x.shape[0])
    ]
).loc[
    :, ['home_team', 'away_team', 'prob_home_win', 'prob_away_win', 'prob_draw']
]


# Step 4: Functionize everything above ----

'''
## EXERCISE 4 ##
In Step 5 we want to validate our predictions. In order to validate our predictions, we need to be
able to use a subset of our data to fit the model so that we can compare those predictions against
out-of-sample results. For this exercise, please functionize everything you do in Steps 2 and 3.
Replace the contents of the function below so that it takes in data and spits out pred (as above).
'''

def train_and_predict(data):
    '''Train a Poisson Bradley-Terry model and produce predictions

    Args:
        data (pandas df): dataframe with cols 'home_team', 'away_team', 'home_goals', 'away_goals'

    Returns:
        pred (pandas df): dataframe with cols 'home_team', 'away_team',
            'prob_home_win', 'prob_away_win', 'prob_draw'
    '''

    pred = pd.DataFrame(
        data={
            'home_team':'Tottenham',
            'away_team':'Arsenal',
            'prob_home_win':1,
            'prob_away_win':0,
            'prob_draw':0
        },
        index=[0]
    )

    return(pred)


# Step 5: Validate predictions ----

import math                                     # for math.log()
from sklearn.model_selection import KFold

'''
To valid our projections, we're going to use k-fold cross-validation. This works by partitioning
our dataset into k equal-sized subsets (called folds). For each fold, we hold out that fold and
train the model using all other folds. Then we evaluate how well our predictions for the held-out
fold compare with the actual results in the held-out fold (hence it's an out-of-sample validation).
'''

kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Set up a dataframe to hold cross-validation results
cv_data = pd.DataFrame()

for train_idx, test_idx in kf.split(data):

    pred = train_and_predict(data.iloc[train_idx])

    test = data.iloc[test_idx]

    # Calculate the log of the predicted probability for the outcome that occurred
    cv_data_k = pd.merge(test, pred, on = ['home_team', 'away_team']).assign(
        '''
        ## EXERCISE 5 ##
        Create a column prob that reflects the predicted probability of the outcome that happened.
        For example, if the home team won and prob_home_win = 0.5, the value of prob would be 0.5.
        '''
        prob=1/3,
        log_prob=lambda x: [math.log(p) for p in x['prob']]
    ).loc[
        :, ['date', 'home_team', 'away_team', 'log_prob']
    ]

    cv_data = pd.concat(objs=[cv_data, cv_data_k])

print(np.mean(cv_data['log_prob']))


# Step 6: Try ridge regression ----

'''
## EXERCISE 6 ##
Now that you've gotten through the whole script, try introducing some regularization to improve
your predictions. To do this, instead of smf.glm().fit(), try smf.glm().fit_regularized().
This fit_regularized() method takes two arguments: alpha and L1_wt. The first argument determines
the amount of regularization (alpha=0 is equivalent to using the fit() method, and alpha=np.inf
forces all coefficients to be exactly zero). For ridge regression, set the second argument to 0.
'''
