
import numpy as np
import pandas as pd
from statsbombpy import sb

match = sb.matches(competition_id=2, season_id=27)      # Premier League 2015/2016

lineup = pd.concat(
    [
        pd.concat(sb.lineups(match_id=x))
            .reset_index(drop=True)
            .rename(columns={'level_0': 'team'})
            .assign(match_id=x)
        for x in match['match_id']
    ]
)

event = pd.concat(
    [sb.events(match_id=x) for x in match['match_id']]
)

match_supplemented = pd.concat(
    [
        match.rename(columns={'home_team': 'team', 'away_score': 'goals_conceded'})[['match_id', 'match_date', 'team', 'goals_conceded']],
        match.rename(columns={'away_team': 'team', 'home_score': 'goals_conceded'})[['match_id', 'match_date', 'team', 'goals_conceded']],
    ]
)

lineup_supplemented = (
    lineup
        .loc[lambda x: [y != [] for y in x['positions']]]
        .assign(
            start=lambda x: [[y['from'] for y in z][0] for z in x['positions']],
            end_raw=lambda x: [([y['to'] for y in z])[-1] for z in x['positions']],
            end=lambda x: np.where([y is None for y in x['end_raw']], '90:00', x['end_raw']),
            start_minutes=lambda x: [int(y[0:2]) for y in x['start']],
            end_minutes=lambda x: [int(y[0:2]) for y in x['end']],
            minutes=lambda x: x['end_minutes'] - x['start_minutes'],
        )
        .merge(match_supplemented, on=['match_id', 'team'], how='left')
        .assign(clean_sheets=lambda x: np.where((x['goals_conceded'] == 0) & (x['minutes'] >= 60), 1, 0))
)

event_supplemented = (
    event
        .assign(
            position=lambda x: np.select(
                condlist=[
                    x['position'].str.contains('Back').fillna(False),
                    x['position'].str.contains('Midfield').fillna(False),
                    x['position'].str.contains('Foward').fillna(False),
                    x['position'].str.contains('Wing').fillna(False),
                    x['position'].str.contains('Goalkeeper').fillna(False),
                ],
                choicelist=['Defender', 'Midfielder', 'Forward', 'Forward', 'Goalkeeper'],
                default=None
            ),
            goals=lambda x: np.where(x['shot_outcome'] == 'Goal', 1, 0),
            assists=lambda x: np.where(x['pass_goal_assist'] == True, 1, 0),
            saves=lambda x: np.where(x['goalkeeper_type'].isin(['Shot Saved', 'Shot Saved Off Target', 'Shot Saved to Post']), 1, 0),
            penalty_saves=lambda x: np.where(x['goalkeeper_type'].isin(['Penalty Save', 'Penalty Saved to Post']), 1, 0),
            penalty_misses=lambda x: np.where((x['shot_type'] == 'Penalty') & (x['shot_outcome'] != 'Goal'), 1, 0),
            yellow_cards=lambda x: np.where(x['bad_behaviour_card'] == 'Yellow Card', 1, 0),
            red_cards=lambda x: np.where(x['bad_behaviour_card'] == 'Red Card', 1, 0),
            own_goals=lambda x: np.where(x['type'] == 'Own Goal Against', 1, 0),
        )
        .rename(columns={'player': 'player_name', 'shot_statsbomb_xg': 'xg'})
)

player_match_summary = (
    event_supplemented
        .groupby(['match_id', 'team', 'player_id', 'player_name', 'position'])
        [['xg', 'goals', 'assists', 'saves', 'penalty_saves', 'penalty_misses', 'yellow_cards', 'red_cards', 'own_goals']]
        .sum()
        .reset_index()
        .merge(lineup_supplemented, on=['match_id', 'team', 'player_id', 'player_name'], how = 'left')
        [[
            'match_id', 'match_date', 'team', 'player_id', 'player_name', 'position', 'minutes',
            'xg', 'goals', 'assists', 'clean_sheets', 'saves', 'penalty_saves', 'penalty_misses',
            'goals_conceded', 'yellow_cards', 'red_cards', 'own_goals'
        ]]
        .assign(player_id=lambda x: [int(y) for y in x['player_id']])
        .sort_values(by=['match_date', 'match_id', 'team', 'player_id', 'position'])
)

player_match_summary.to_csv('~/Downloads/player_match_summmary_epl_2015-2016.csv', index = False)
