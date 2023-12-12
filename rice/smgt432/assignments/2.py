
import numpy as np
import pandas as pd
from statsbombpy import sb

match = sb.matches(competition_id=2, season_id=27)      # Premier League 2015/2016

# Extract passes to infer player's preferred foot
passes = pd.concat(
    [sb.events(match_id=x).loc[lambda x: x['type'] == "Pass"] for x in match['match_id']]
)

# Define player's preferred foot to be the one they use for the majority of their passes
player_pref_foot = (
    passes
        .groupby('player_id')['pass_body_part']
        .apply(lambda x: (x == 'Right Foot').mean())
        .reset_index()
        .assign(pref_foot=lambda x: np.where(x['pass_body_part'] > 0.5, 'Right Foot', 'Left Foot'))
        .loc[:, ['player_id', 'pref_foot']]
)

shot = pd.concat(
    [sb.events(match_id=x).loc[lambda x: x['type'] == "Shot"] for x in match['match_id']]
)

shot_clean = (
    shot.loc[lambda x: x['type'] == 'Shot']
        .reset_index()
        .merge(player_pref_foot, on='player_id', how='left')
        [[
            'id',
            'shot_outcome',
            'period',
            'minute',
            'second',
            'location',
            'team_id',
            'team',
            'player_id',
            'player',
            'position',
            'pref_foot',
            'play_pattern',
            'shot_type',
            'shot_body_part',
            'shot_technique',
            'under_pressure',
            'shot_first_time',
            'shot_aerial_won',
            'shot_one_on_one',
            'shot_freeze_frame'
        ]]
)

shot_shuffled = shot_clean.sample(frac=1, random_state=42)
split_index = 6908
shot_train = shot_shuffled[:split_index]
shot_test = shot_shuffled[split_index:].drop(['shot_outcome'], axis=1)

shot_train.to_csv('~/Downloads/shot_train.csv', index=False)
shot_test.to_csv('~/Downloads/shot_test.csv', index=False)
