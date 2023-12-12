

# Step 1: Read the data into Python ----

import pandas as pd

data_raw = pd.read_csv('~/git/teaching/rice/smgt432/data/super_league_2022.csv')

# Extract integer home_goals and away_goals from score string
data = data_raw.assign(
    home_goals=data_raw['score'].str[0].astype(int),
    away_goals=data_raw['score'].str[2].astype(int)
)

# Extract integer home_goals and away_goals from score string
data = data_raw.assign(
    home_goals=data_raw['xg_home'],
    away_goals=data_raw['xg_away']
)


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

# Fit the Poisson Bradley-Terry model
poisson_model = smf.glm(
    formula="goals ~ home + offense + defense",
    data=goal_model_data,
    family=sm.families.Poisson()
).fit()


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
    prob_home_win=lambda x: [
        # Calculate the probability that the Skellam-distributed difference is greater than zero
        sum(skellam.pmf(range(1, 10), x['pred_goals_home'][i], x['pred_goals_away'][i])) for i in range(0, x.shape[0])
    ],
    prob_away_win=lambda x: [
        # Calculate the probability that the Skellam-distributed difference is less than zero
        sum(skellam.pmf(range(-9, 0), x['pred_goals_home'][i], x['pred_goals_away'][i])) for i in range(0, x.shape[0])
    ],
    prob_draw=lambda x: [
        # Calculate the probability that the Skellam-distributed difference is exactly zero
        skellam.pmf(0, x['pred_goals_home'][i], x['pred_goals_away'][i]) for i in range(0, x.shape[0])
    ]
).loc[
    :, ['home_team', 'away_team', 'prob_home_win', 'prob_away_win', 'prob_draw']
]


# Step 4: Functionize everything above ----

def train_and_predict(data):
    '''Train a Poisson Bradley-Terry model and produce predictions

    Args:
        data (pandas df): dataframe with cols 'home_team', 'away_team', 'home_goals', 'away_goals'

    Returns:
        pred (pandas df): dataframe with cols 'home_team', 'away_team',
            'prob_home_win', 'prob_away_win', 'prob_draw'
    '''

    # Wrangle data for the Bradley-Terry model. We need two rows for every game, one for goals
    # scored by the home team and one for goals scored by the away team.
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

    # Fit the Poisson Bradley-Terry model
    poisson_model = smf.glm(
        formula="goals ~ home + offense + defense",
        data=goal_model_data,
        family=sm.families.Poisson()
    ).fit_regularized(alpha = math.exp(-5.3), L1_wt = 0)

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
        prob_home_win=lambda x: [
            # Calculate the probability that the Skellam-distributed difference is greater than zero
            sum(skellam.pmf(range(1, 10), x['pred_goals_home'][i], x['pred_goals_away'][i])) for i in range(0, x.shape[0])
        ],
        prob_away_win=lambda x: [
            # Calculate the probability that the Skellam-distributed difference is less than zero
            sum(skellam.pmf(range(-9, 0), x['pred_goals_home'][i], x['pred_goals_away'][i])) for i in range(0, x.shape[0])
        ],
        prob_draw=lambda x: [
            # Calculate the probability that the Skellam-distributed difference is exactly zero
            skellam.pmf(0, x['pred_goals_home'][i], x['pred_goals_away'][i]) for i in range(0, x.shape[0])
        ]
    ).loc[
        :, ['home_team', 'away_team', 'prob_home_win', 'prob_away_win', 'prob_draw']
    ]

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
        # Determine which outcome occurred
        home_win=lambda x: (x['home_goals'] > x['away_goals']),
        away_win=lambda x: (x['home_goals'] < x['away_goals']),
        draw=lambda x: (x['home_goals'] == x['away_goals']),
        # Get the log-probability of the event that actually happened
        prob=lambda x: x['home_win'] * x['prob_home_win'] + x['away_win'] * x['prob_away_win'] + x['draw'] * x['prob_draw'],
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
