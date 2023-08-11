"""
benchmark stuff for now
"""
import pandas as pd
import numpy as np


def predict_thrombolysis_rate_in_benchmark_scenario(
    big_data, model, benchmark_team_id_list, team_id_column='stroke_team_id', prediction_column='thrombolysis'):
    """
    wrapper for the other one
    """
    all_team_names = sorted(list(set(big_data[team_id_column])))
    
    all_true_rates = []
    all_base_rates = []
    all_benchmark_rates = []
    
    for team_id in all_team_names:

        # Data for all patients at this hospital
        team_big_df_orig = big_data[big_data[team_id_column] == team_id].copy()
        
        # Find the true thrombolysis rate:
        true_thrombolysis_rate = team_big_df_orig[prediction_column].mean()
        all_true_rates.append(true_thrombolysis_rate)
        
        # Drop the thrombolysis result:
        team_big_df_orig = team_big_df_orig.drop(prediction_column, axis=1).copy()
        # Now the input dataframe should have the same columns as the
        # data that was used to train the model.
        
        # Predict for this team:        
        team_big_df_orig[team_id_column] = team_big_df_orig[
            team_id_column].astype('category')
        base_rate = model.predict(team_big_df_orig).mean()
        all_base_rates.append(base_rate)
        
        # Predict for all benchmark teams:
        predictions_for_this_team = predict_benchmark_thrombolysis(
            model, 
            team_big_df_orig, 
            benchmark_team_id_list, 
            stroke_team_id_column=team_id_column
            )
        all_benchmark_rates.append(
            predictions_for_this_team['majority_vote'].mean())
        
    all_teams_results = pd.DataFrame(
        data=np.vstack((all_true_rates, all_base_rates, all_benchmark_rates)).T,
        index=all_team_names,
        columns=['True', 'Base', 'Benchmark'],
        dtype=float
        )
    return all_teams_results

    
def predict_benchmark_thrombolysis(
        model,
        team_big_df_orig,
        benchmark_team_id_list,
        stroke_team_id_column='stroke_team_id',
        prediction_column='thrombolysis'
        ):
    """
    Predict thrombolysis rate according to benchmark teams.
    
    model - trained model for predicto
    team_big_df - all data for the hospital we're testing
    benchmark_team_id_list - what we change the stroke team id to for the benchmark teams
    """
    # Sanity checks
    # Does the input dataframe contain the named columns for
    # stroke team ID and predicted quantity?
    # And does it contain all of the features that are expected
    # by the model?
    data_columns = list(team_big_df_orig.columns)
    feature_names = model.feature_name_
    err_str = ''
    err_count = 1
    if stroke_team_id_column not in data_columns:
        err_str += ''.join([
            f'Problem {err_count}: ',
            f'Expected a column named {stroke_team_id_column} ',
            'to contain the stroke team IDs. ',
            'You can set the expected column name with the kwarg ',
            '"stroke_team_id_column".\n'
            ])
        err_count += 1
    else:
        pass
    
    # Check that all expected feature names are present in the data.
    # The order of the features doesn't matter, just that they
    # exist.
    for feature in feature_names:
        if feature not in data_columns:
            err_str += ''.join([
                f'Problem {err_count}: ',
                f'Expected a column named {feature}.\n'
                ])
            err_count += 1
        else:
            pass
            
    # Print a warning message if necessary:
    if len(err_str) > 0:
        raise KeyError(err_str)
    else:
        pass

    
    # Store results for all benchmark teams in here:
    all_benchmark_predictions = pd.DataFrame()
    
    for benchmark_team_id in benchmark_team_id_list:
        # Make a copy of the original hospital data
        team_big_df = team_big_df_orig.copy()
        # Update the stroke team ID to match the benchmark hospital ID:
        team_big_df[stroke_team_id_column] = benchmark_team_id
        # This conversion must go after the hospital ID is updated:
        team_big_df[stroke_team_id_column] = team_big_df[
            stroke_team_id_column].astype('category')

        # Get predictions
        predicted = model.predict(team_big_df)

        # Store the results in the big results dataframe:
        all_benchmark_predictions[benchmark_team_id] = predicted

    # Find majority decision.    
    # How many teams must say "yes" for the majority to be "yes"?
    # e.g. for 25 teams, this value is 13.
    bench_majority_threshold = int(np.ceil(0.5 * len(benchmark_team_id_list)))
    # Add a column for the majority vote.
    all_benchmark_predictions['majority_vote'] = (
        all_benchmark_predictions.sum(axis=1)
        // bench_majority_threshold
        )
    # ^ the // means that when the number of teams answering "yes" is 
    # at least the majority threshold, the value is 1 and when the
    # number of teams is less than the threshold, the value is 0.
    return all_benchmark_predictions
