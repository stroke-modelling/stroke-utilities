"""
benchmark stuff for now
"""
import pandas as pd
import numpy as np
from classes.pathway import SSNAP_Pathway
from math import sqrt
from scipy import stats
from .process_data import one_hot_encode_column


def create_masks(
        group_df,
        time_left_after_scan_mins: np.array,
        time_to_treatment_mins: np.array,        
        limit_mins: float=np.inf,
        minutes_left: float=15.0,
        col_precise_onset_known: str='precise_onset_known',
        col_onset_to_arrival: str='onset_to_arrival_time',
        col_arrival_to_scan: str='arrival_to_scan_time',
        ):
    """
    Make masks of whether patients meet various conditions.

    The masks match the diagram in the 
    extract_hospital_performance() docstring.

    Inputs:
    -------
    time_left_after_scan_mins - array. One value per patient for
                                time left after the scan for 
                                treatment.
    time_to_treatment_mins    - array. One value per patient for
                                time from arrival to treatment.
    limit_mins                - float. The time limit that the 
                                times in each step are compared 
                                with in creating the mask.
    minutes_left              - float. How much time there must be
                                left after scan for treatment to be
                                considered.
    col_precise_onset_known   - str. Name of the column containing
                                whether precise onset is known.
    col_onset_to_arrival      - str. Name of the column containing
                                onset to arrival time.
    col_arrival_to_scan       - str. Name of the column containing
                                arrival to scan time.

    Returns:
    --------
    masks_dict - dictionary. Contains the masks.
    """
    # Create masks.
    # 1: Onset time known
    mask1 = group_df[col_precise_onset_known] == 1
    # 2: Mask 1 and onset to arrival on time
    mask2 = ((mask1 == True) & 
             (group_df[col_onset_to_arrival] <= limit_mins))
    # 3: Mask 2 and arrival to scan on time
    mask3 = (
        (mask2 == True) & 
        (group_df[col_arrival_to_scan] <= limit_mins)
        )
    # 4: Mask 3 and onset to scan on time
    mask4 = (
        (mask3 == True) &
        ((group_df[col_onset_to_arrival] + 
          group_df[col_arrival_to_scan]) <= limit_mins)
        )
    # 5: Mask 4 and enough time to treat
    mask5 = ((mask4 == True) &
             (time_left_after_scan_mins >= minutes_left))
    # 6: Mask 5 and treated
    mask6 = ((mask5 == True) & 
             (time_to_treatment_mins >= 0))

    masks_dict = dict(
        mask1_all_onset_known=mask1,
        mask2_mask1_and_onset_to_arrival_on_time=mask2,
        mask3_mask2_and_arrival_to_scan_on_time=mask3, 
        mask4_mask3_and_onset_to_scan_on_time=mask4, 
        mask5_mask4_and_enough_time_to_treat=mask5, 
        mask6_mask5_and_treated=mask6
        )
    return masks_dict


def calculate_more_times_for_dataframe(group_df):
    """
    Combine the existing time data into more time measures.

    Creates:
    + scan to needle time
    + scan to puncture time
    + time left for needle
    + time left for puncture

    Inputs:
    -------
    group_df - pandas DataFrame. Stores the original time data
               that will be combined into new measures.

    Returns:
    --------
    group_df - pandas DataFrame. Same as the input dataframe but 
               with the new time arrays added.                     
    """
    # Scan to treatment
    scan_to_needle_mins = (
        group_df['ArrivaltoThrombolysisMinutes'] -
        group_df['ArrivaltoBrainImagingMinutes']
    )
    scan_to_puncture_mins = (
        group_df['ArrivaltoArterialPunctureMinutes'] -
        group_df['ArrivaltoBrainImagingMinutes']
    )
    # Replace any zero scan to treatment times with 1 (for log?)
    # and store in the dataframe.
    scan_to_needle_mins[scan_to_needle_mins == 0] = 1
    group_df['scan_to_needle_mins'] = scan_to_needle_mins
    scan_to_puncture_mins[scan_to_puncture_mins == 0] = 1
    group_df['scan_to_puncture_mins'] = scan_to_puncture_mins


    # Time left after scan for thrombolysis...
    group_df['time_left_for_ivt_after_scan_mins'] = np.maximum((
        allowed_onset_to_needle_time_mins -
        (group_df['OnsettoArrivalMinutes'] + 
          group_df['ArrivaltoBrainImagingMinutes'])
        ), -0.0)
    # ... and thrombectomy:
    group_df['time_left_for_mt_after_scan_mins'] = np.maximum((
        allowed_onset_to_puncture_time_mins -
        (group_df['OnsettoArrivalMinutes'] + 
          group_df['ArrivaltoBrainImagingMinutes'])
        ), -0.0)
    # If the time is negative, set it to -0.0.
    # The minus distinguishes the clipped values from the ones
    # that genuinely have exactly 0 minutes left.
    return group_df


def calculate_proportions(masks_dict: dict):
    """
    Find proportions of patients who answer True to each mask.
    
    The proportion is out of those who answered True to the 
    previous mask, not out of the whole cohort.
    
    Inputs:
    -------
    masks_dict - a dictionary containing the masks described in
                 the docstring of extract_hospital_performance().
    
    Returns:
    --------
    proportions_dict - a dictionary containing the proportions
                       of patients answering True to each mask.
                       The keys are named similarly to the input
                       mask dictionary.
    """
    # Store results in here:
    proportions_dict = dict()
    # Get the mask names assuming that the masks are stored in
    # the dictionary in order.
    mask_names = list(masks_dict.keys())
    for j, mask_name_now in enumerate(mask_names):
        mask_now = masks_dict[mask_name_now]
        if j > 0:
            # If there's a previous mask, find it from the dict:
            mask_before = masks_dict[mask_names[j-1]]
            mask_name_before = f'mask{j}'
        else:
            # All patients answered True in the previous step.
            mask_before = np.full(len(mask_now), 1)
            mask_name_before = 'all'
        
        # Proportion is Yes to Mask now / Yes to Mask before.
        proportion = (np.sum(mask_now) / np.sum(mask_before)
                      if np.sum(mask_before) > 0 else np.NaN)
        # Create a name string for this proportion.
        # Replace the initial "maskX_" with "proportionX_":
        proportion_name = f'proportion{j+1}_of_{mask_name_before}_with'
        # Remove previous mask name and "_and" from the string:
        p = '_'.join(mask_name_now[6:].split(mask_name_before)[-1:])
        p = '_and'.join(p.split('_and')[-1:])
        proportion_name += p
    
        # Store result with a similar name to the original mask.
        proportions_dict[proportion_name] = proportion
    return proportions_dict


def calculate_lognorm_parameters(group_df, input_dicts):
    """ 
    Calculate parameters of lognorm time distributions.
    
    Inputs:
    -------
    group_df    - pandas DataFrame. Stores the original time data
                  that will be lognorm-ed and analysed.
    input_dicts - list of dicts. Stores the instructions for which
                  times to pull out of the dataframe and what to 
                  name the resulting data. Each dict must contain:
                  + label: str. name for the resulting data.
                  + mask: array. Mask of True/False for the time
                    array so that only certain times are used in
                    the calculations.
                  + column: str. The name of the column of times
                    in the dataframe.
    
    Returns:
    --------
    results_dict - dict. Contains a mu (mean) and sigma (standard
                   deviation) for the lognorm distributions of the
                   times selected by each of the input dicts.
    """
    # Place all of the results in here:
    results_dict = dict()
    for d in input_dicts:
        # Pick out the times from the chosen column and use the
        # chosen mask to only select a subset from the column.
        times_not_logged = group_df[d['mask']][d['column']].copy()
        # Set times of exactly zero to a small value to prevent
        # errors when logging it:
        times_not_logged[times_not_logged == 0.0] = 1 # minutes
        times = np.log(times_not_logged)
        if len(times) > 10:
            # Calculate the lognorm mu and sigma
            mu = times.mean()
            sigma = times.std()
        else:
            # If there are too few patients, don't record values.
            mu = np.NaN
            sigma = np.NaN
        # Store the values in the results dictionary.
        results_dict['lognorm_mu_' + d['label']] = mu
        results_dict['lognorm_sigma_' + d['label']] = sigma
    return results_dict


def predict_thrombolysis_rate_in_benchmark_scenario(
        big_data, 
        model, 
        benchmark_team_id_list, 
        features_to_model,
        limit_treatment_mins,
        minutes_left,
        all_team_names,
        team_id_column='stroke_team_id', 
        prediction_column='thrombolysis',
        split_by_stroke_type=False
    ):
    """
    Wrapper for predict_benchmark_thrombolysis() for multiple teams.

    Predict the thrombolysis rate across all stroke teams if each one
    used a benchmark majority vote of whether or not to treat patients.

    Inputs:
    -------
    big_data               - pd.DataFrame. Full data for all patients 
                             and stroke teams.
    model                  - e.g. LightGBMClassifier. Trained model.
    benchmark_team_id_list - list. List of benchmark team IDs.
    features_to_model      - list of str. Column names to restrict the
                             big dataframe to so that the data is ready
                             to be passed to the model.            
    limit_treatment_mins   - float.
    minutes_left           - float.
    all_team_names         - list. List of stroke team names.
    team_id_column         - str. Name of the stroke team column.
    prediction_column      - str. Name of the predicted value column.
    split_by_stroke_type   - bool.

    Returns:
    --------
    all_team_results - pandas DataFrame. For each stroke team, the
                       real life rate, predicted rate based on real
                       patients, and predicted rate based on benchmark
                       majority decisions.
    """
    if split_by_stroke_type:
        stroke_type_list = ['nlvo', 'lvo', 'other', 'mixed']
    else:
        stroke_type_list = ['mixed']
    
    all_index_names = []
    all_stroke_types = []
    
    all_base_rates = []
    all_benchmark_rates = []
    all_base_rates_of_mask5 = []
    all_benchmark_rates_of_mask5 = []
    
    for team_id in all_team_names:

        team_id_column = f'team_{team_id}'
        # Data for all patients at this hospital
        team_big_df_orig = big_data[big_data[team_id_column] == 1].copy()

        stroke_type_mask_dict = {
            'lvo': ((team_big_df_orig['infarction']==1) & 
                    (team_big_df_orig['stroke_severity']>=11)),
            'nlvo': ((team_big_df_orig['infarction']==1) & 
                     (team_big_df_orig['stroke_severity']<10)),
            'other': (team_big_df_orig['infarction']!=1),
            'mixed': ([True] * len(team_big_df_orig))
        }
        
        for stroke_type in stroke_type_list:
            all_index_names.append(f'{team_id}')
            all_stroke_types.append(stroke_type)

            if split_by_stroke_type:
                # Mask by stroke type:
                team_big_df = team_big_df_orig[
                    stroke_type_mask_dict[stroke_type]]
            else:
                team_big_df = team_big_df_orig.copy()
            n_with_this_stroke_type = len(team_big_df)

            if n_with_this_stroke_type == 0:
                # Prevent division by zero.
                base_thrombolysis_rate = np.NaN
                benchmark_thrombolysis_rate = np.NaN
                base_thrombolysis_rate_of_mask5 = np.NaN
                benchmark_thrombolysis_rate_of_mask5 = np.NaN
            else:
                # Mask to only get eligible patients:
                masks_dict_ivt = create_masks(
                    team_big_df,
                    team_big_df['time_left_for_ivt_after_scan_mins'],
                    team_big_df['arrival_to_thrombolysis_time'],
                    limit_treatment_mins,
                    minutes_left
                    )
                team_big_df = team_big_df[
                    masks_dict_ivt['mask5_mask4_and_enough_time_to_treat']]
        
                # Restrict to only the features used by the model:
                team_big_df = team_big_df[features_to_model]
        
                # Find the base thrombolysis rate:
                base_n_thrombolysed = team_big_df[
                    prediction_column].sum()
                base_thrombolysis_rate_of_mask5 = team_big_df[
                    prediction_column].mean()
                base_thrombolysis_rate = (
                    base_n_thrombolysed /
                    n_with_this_stroke_type
                    )
                
                # Drop the thrombolysis result:
                team_big_df = team_big_df.drop(
                    prediction_column, axis=1).copy()
                # Now the input dataframe should have the same columns as the
                # data that was used to train the model.
                
                # Predict for all benchmark teams:
                predictions_for_this_team = predict_benchmark_thrombolysis(
                    model, 
                    team_big_df, 
                    benchmark_team_id_list, 
                    stroke_team_id_column=team_id_column
                    )
                benchmark_n_thrombolysed = (
                    predictions_for_this_team['majority_vote'].sum())
                benchmark_thrombolysis_rate_of_mask5 = (
                    predictions_for_this_team['majority_vote'].mean())
                benchmark_thrombolysis_rate = (
                    benchmark_n_thrombolysed / 
                    n_with_this_stroke_type
                    )

            # Store results in big lists:
            all_base_rates.append(base_thrombolysis_rate)
            all_benchmark_rates.append(benchmark_thrombolysis_rate)
            all_base_rates_of_mask5.append(base_thrombolysis_rate_of_mask5)
            all_benchmark_rates_of_mask5.append(benchmark_thrombolysis_rate_of_mask5)

    # Convert multiple results lists to one results dataframe:
    all_teams_results = pd.DataFrame(
        data=np.vstack(
            (all_stroke_types,
             all_base_rates,
             all_benchmark_rates,
             all_base_rates_of_mask5,
             all_benchmark_rates_of_mask5)
            ).T,
        index=all_index_names,
        columns=[
            'Stroke type',
            'Base',
            'Benchmark',
            'Base rate of mask5',
            'Benchmark rate of mask5'
            ],
        ).astype(dtype={
            'Stroke type':str,
            'Base':float,
            'Benchmark':float,
            'Base rate of mask5':float,
            'Benchmark rate of mask5':float
            })
    all_teams_results.index.name = 'stroke_team_id'
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

    Inputs:
    -------
    model                  - e.g. LightGBMClassifier. Trained model.
    team_big_df_orig       - pd.DataFrame. Full data for all patients 
                             at this stroke team.
    benchmark_team_id_list - list. List of benchmark team IDs.
    stroke_team_id_column  - str. Name of the stroke team column.
    prediction_column      - str. Name of the predicted value column.

    Returns:
    --------
    all_benchmark_predictions - pandas DataFrame. Each patient's 
        predicted thrombolysis decision for each of the benchmark teams
        and the benchmark majority decision.
    
    """
    # Sanity checks
    # Does the input dataframe contain the named columns for
    # stroke team ID and predicted quantity?
    # And does it contain all of the features that are expected
    # by the model?
    data_columns = list(team_big_df_orig.columns)
    feature_names = model.get_booster().feature_names 
    # LightGBM: model.feature_name_
    err_str = ''
    err_count = 1
    # if stroke_team_id_column not in data_columns:
    #     err_str += ''.join([
    #         f'Problem {err_count}: ',
    #         f'Expected a column named {stroke_team_id_column} ',
    #         'to contain the stroke team IDs. ',
    #         'You can set the expected column name with the kwarg ',
    #         '"stroke_team_id_column".\n'
    #         ])
    #     err_count += 1
    # else:
    #     pass
    
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

    team_big_df_orig[stroke_team_id_column] = 0# benchmark_team_id
    for benchmark_team_id in benchmark_team_id_list:
        # Make a copy of the original hospital data
        team_big_df = team_big_df_orig.copy()
        # Update the stroke team ID to match the benchmark hospital ID:
        benchmark_team_id_column = f'team_{benchmark_team_id}'
        team_big_df[benchmark_team_id_column] = 1
        # # For LightGBM where stroke team is a category:
        # # This conversion must go after the hospital ID is updated:
        # team_big_df[stroke_team_id_column] = team_big_df[
        #     stroke_team_id_column].astype('category')

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


def extract_hospital_performance(
        stroke_team: str,
        stroke_type: str,
        group_df: pd.DataFrame,
        limit_ivt_mins: float,
        limit_mt_mins: float,
        minutes_left: float
        ):
    """ 
    Measure metrics of the hospital's performance.
    
    The metrics are measured on various subgroups of patients that
    are recorded using masks. Each patient has a value of True when
    the mask conditions are met and False when they are not.
    The time distribution metrics are calculated from the log-normal
    distribution of times.
    
    The measured values are:
    + admissions per year
    + proportion of all patients given thrombolysis
    + proportion of all patients given thrombectomy
    + proportion of patients given thrombectomy 
      who were also given thrombolysis
    For thrombolysis and for thrombectomy:
    + proportion of all patients with known onset time
    + proportion of mask1 with onset to arrival on time 
    + proportion of mask2 with arrival to scan on time 
    + proportion of mask3 with onset to scan on time 
    + proportion of mask4 with enough time to treat 
    + proportion of mask5 with treated 
    For subgroups of patients meeting certain conditions
    and for thrombolysis and for thrombectomy:
    + mean (mu) of lognormed onset to arrival times 
    + standard deviation (sigma) of lognormed onset to arrival times
    + mean (mu) of lognormed scan to arrival times
    + standard deviation (sigma) of lognormed scan to arrival times
    + mean (mu) of lognormed scan to treatment times
    + standard deviation (sigma) of lognormed scan to treatment times
    
    ----- Method -----
    The masks are created in the following way. With each step, whittle
    down the full group of patients. In the example, the sizes of 
    blocks are arbitrary.     
    Key:
    ░ - patients still in the subgroup
    ▒ - patients rejected from the subgroup at this step
    █ - patients rejected from the subgroup in previous steps

    ▏Start: Full group                                                ▕
    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
    ▏-------------------------All patients----------------------------▕
    ▏                                                                 ▕
    ▏Mask 1: Is onset time known?                                     ▕
    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    ▏--------------------Yes----------------------▏---------No--------▕
    ▏                                             ▏                   ▕
    ▏Mask 2: Is onset to arrival within the time limit?               ▕
    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒█████████████████████
    ▏---------------Yes----------------▏----No----▏------Rejected-----▕
    ▏                                  ▏          ▏                   ▕
    ▏Mask 3: Is arrival to scan wtihin the time limit?                ▕
    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒████████████████████████████████
    ▏------------Yes------------▏--No--▏-----------Rejected-----------▕
    ▏                           ▏      ▏                              ▕
    ▏Mask 4: Is onset to scan within the time limit?                  ▕
    ░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒███████████████████████████████████████
    ▏----------Yes---------▏-No-▏---------------Rejected--------------▕
    ▏                      ▏    ▏                                     ▕
    ▏Mask 5: Is there enough time left for thrombolysis/thrombectomy? ▕
    ░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒████████████████████████████████████████████
    ▏------Yes------▏--No--▏------------------Rejected----------------▕
    ▏               ▏      ▏                                          ▕
    ▏Mask 6: Did the patient receive thrombolysis/thrombectomy?       ▕
    ░░░░░░░░░░░▒▒▒▒▒███████████████████████████████████████████████████
    ▏----Yes---▏-No-▏---------------------Rejected--------------------▕


    Patient proportions measured:    
    +--------------------------------+--------------------------------+
    | Proportion                     | Measure                        |
    +--------------------------------+--------------------------------+
    | Thrombolysis rate or           | Total number treated           |
    | thrombectomy rate              | divided by all patients.       |
    | Onset known                    | "Yes" to Mask 1 divided by     |
    |                                | all patients.                  |
    | Onset to arrival within limit  | "Yes" to Mask 2 divided by     |
    |                                | "Yes" to Mask 1.               |
    | Arrival to scan within limit   | "Yes" to Mask 3 divided by     |
    |                                | "Yes" to Mask 2.               |
    | Onset to scan within limit     | "Yes" to Mask 4 divided by     |
    |                                | "Yes" to Mask 3.               |
    | Enough time left for treatment | "Yes" to Mask 5 divided by     |
    |                                | "Yes" to Mask 4.               |
    | "Chosen" for thrombolysis      | "Yes" to Mask 6 divided by     |
    | or for thrombectomy            | "Yes" to Mask 5.               |
    +--------------------------------+--------------------------------+
    The "proportion chosen for thrombolysis" is a different measure
    from the "thrombolysis rate", which is the proportion of all of the
    patients at the start who were given thrombolysis. It is possible
    some patients received thrombolysis in real life but that by this
    process they were rejected before Mask 6.

    The log-normal mean and standard deviation (mu and sigma) are taken
    for the groups of patients who answer "Yes" to everything up to and
    including particular steps.
    +---------------------------------+-------------------------------+
    | Subgroup who answer "yes" at... | Log-normal distribution       |
    +---------------------------------+-------------------------------+
    |                          Mask 2 | Onset to arrival time         |
    |                          Mask 3 | Arrival to scan time          |
    |                          Mask 6 | Scan to needle or             |
    |                                 | scan to puncture time         |
    +---------------------------------+-------------------------------+
    
    Thrombolysis and thrombectomy can be given different time limits
    for the creation of these masks.
    
    Inputs:
    -------
    stroke_team    - str. Name of the hospital for labelling.
    stroke_type    - str. Names of the stroke types in this data (i.e.
                     non-Large Vessel Occlusion (nLVO), Large Vessel 
                     Occlusion (LVO), other).
    group_df       - pandas DataFrame. Contains all the hospital data.
    limit_ivt_mins - float. Cutoff time for thrombolysis.
    limit_mt_mins  - float. Cutoff time for thrombectomy.
    minutes_left   - float. Minutes allowed between scan and 
                     thrombolysis cutoff.
    
    Returns:
    --------
    performance_dict - dictionary. Contains various metrics of hospital
                       performance. 
    """
    # Record admission numbers
    admissions = group_df.shape[0]

    # Calculate more times from the existing data:
    # group_df = calculate_more_times_for_dataframe(group_df)

    # Find the proportion of the whole cohort that receives
    # treatment.
    proportion_all_ivt = (group_df['thrombolysis'] == 1).mean()
    proportion_all_mt = (
        group_df['arrival_to_thrombectomy_time'] >= 0).mean()
    proportion_mt_also_receiving_ivt = (
        np.sum((group_df['thrombolysis'] == 1) & 
               (group_df['arrival_to_thrombectomy_time'] >= 0)) /
        np.sum(group_df['arrival_to_thrombectomy_time'] >= 0)
        if np.sum(group_df['arrival_to_thrombectomy_time'] >= 0) > 0 
        else np.NaN  # Prevent division by zero if no patients have MT.
        )

    # ----- Thrombolysis -----
    # Masks of patients who answer True to each step:
    masks_dict_ivt = create_masks(
        group_df,
        group_df['time_left_for_ivt_after_scan_mins'],
        group_df['arrival_to_thrombolysis_time'],
        limit_ivt_mins,
        minutes_left
        )
    # Proportion of patients in each mask:
    proportions_dict_ivt = calculate_proportions(masks_dict_ivt)
    # Record the mu and sigma for certain times and subgroups.
    # Set up with these dictionaries:
    dicts_ivt = [
        dict(label = 'onset_arrival_mins',
             mask = masks_dict_ivt[
                'mask2_mask1_and_onset_to_arrival_on_time'],
             column = 'onset_to_arrival_time'
             ),
        dict(label = 'arrival_scan_arrival_mins',
             mask = masks_dict_ivt[
                'mask3_mask2_and_arrival_to_scan_on_time'],
             column = 'arrival_to_scan_time'
             ),
        dict(label = 'scan_needle_mins',
             mask = masks_dict_ivt['mask6_mask5_and_treated'],
             column = 'scan_to_thrombolysis_time'
             )
        ]
    lognorm_dict_ivt = calculate_lognorm_parameters(group_df, dicts_ivt)


    # ----- Thrombectomy -----
    # Masks of patients who answer True to each step:
    masks_dict_mt = create_masks(
        group_df,
        group_df['time_left_for_mt_after_scan_mins'],
        group_df['arrival_to_thrombectomy_time'],
        limit_mt_mins,
        minutes_left
        )
    # Proportion of patients in each mask:
    proportions_dict_mt = calculate_proportions(masks_dict_mt)
    # Record the mu and sigma for certain times and subgroups.
    # Set up with these dictionaries:
    dicts_mt = [
        dict(label = 'onset_arrival_mins',
             mask = masks_dict_mt[
                'mask2_mask1_and_onset_to_arrival_on_time'],
             column = 'onset_to_arrival_time'
             ),
        dict(label = 'arrival_scan_arrival_mins',
             mask = masks_dict_mt[
                'mask3_mask2_and_arrival_to_scan_on_time'],
             column = 'arrival_to_scan_time'
             ),
        dict(label = 'scan_puncture_mins',
             mask = masks_dict_mt['mask6_mask5_and_treated'],
             column = 'scan_to_thrombectomy_time'
             )
        ]
    lognorm_dict_mt = calculate_lognorm_parameters(group_df, dicts_mt)

    # ----- Combine results -----
    performance_dict = dict()
    performance_dict['stroke_team_id'] = stroke_team
    performance_dict['stroke_type'] = stroke_type
    performance_dict['admissions'] = admissions
    performance_dict['proportion_of_all_with_ivt'] = proportion_all_ivt
    performance_dict['proportion_of_all_with_mt'] = proportion_all_mt
    performance_dict['proportion_of_mt_with_ivt'] = \
        proportion_mt_also_receiving_ivt

    # Take these dictionaries from earlier...
    dicts_to_combine = [proportions_dict_ivt, lognorm_dict_ivt,
                        proportions_dict_mt, lognorm_dict_mt]
    # ... and merge them into the new results dictionary.
    for i, d in enumerate(dicts_to_combine):
        # Add extra label to prevent repeat keys in the combined dict.
        extra_label = '_ivt' if i < 2 else '_mt'
        keys = list(d.keys())
        for key in keys:
            performance_dict[key + extra_label] = d[key]

    
    # Recalculate treatment rates to reflect the changes.
    # n.b. if this calculation is done for "base" scenario,
    # it will return different answers than the start 
    # values. This is because in the real data, some 
    # patients do not meet all of these conditions yet
    # still receive treatment.
    rates = []
    for treatment in ['ivt', 'mt']:
        rate = 1.0
        keys = [
            'proportion1_of_all_with_onset_known_',
            'proportion2_of_mask1_with_onset_to_arrival_on_time_',
            'proportion3_of_mask2_with_arrival_to_scan_on_time_',
            'proportion4_of_mask3_with_onset_to_scan_on_time_',
            'proportion5_of_mask4_with_enough_time_to_treat_',
            'proportion6_of_mask5_with_treated_'
            ]
        for key in keys:
            rate *= performance_dict[key + treatment] 
        rates.append(rate)
    performance_dict['proportion_of_all_with_mask6_and_ivt'] = rates[0]
    performance_dict['proportion_of_all_with_mask6_and_mt'] = rates[1]

    return group_df, performance_dict, masks_dict_ivt, masks_dict_mt


def build_scenario_hospital_performance(
        hospital_performance,
        stroke_team,
        scenario_vals_dict,
        onset_time_known_proportion_dict,
        df_benchmark_codes
        ):
    """
    Create the scenario parameters for all stroke types and scenarios.

    Inputs:
    -------
    hospital_performance             - pd.DataFrame. The output from
                                       extract_hospital_performance().
    stroke_team                      - str. Name of this stroke team.
    scenario_vals_dict               - dict. Speed scenario parameters.
    onset_time_known_proportion_dict - dict. Onset scenario parameters.
    df_benchmark_codes               - pd.DataFrame. Contains benchmark
                                       thrombolysis rates for all 
                                       teams.

    Returns:
    --------
    df_performance_scenarios - pd.DataFrame. Contains the pathway
                               parameters for all scenarios and stroke
                               types for this team.
    """
    # Scenario changes to the performance data
    scenario_dicts = [
        dict(speed=0, onset=0, benchmark=0),  # base
        dict(speed=0, onset=0, benchmark=1),  # benchmark
        dict(speed=0, onset=1, benchmark=0),  # onset
        dict(speed=1, onset=0, benchmark=0),  # speed
        dict(speed=0, onset=1, benchmark=1),  # onset + benchmark
        dict(speed=1, onset=0, benchmark=1),  # speed + benchmark
        dict(speed=1, onset=1, benchmark=0),  # speed + onset
        dict(speed=1, onset=1, benchmark=1),  # speed + onset + benchmark
        ]

    count = 0
    for d in scenario_dicts:
        for stroke_type in ['lvo', 'nlvo', 'other', 'mixed']:
            
            # Gather the relevant hospital data:
            hospital_data_original = hospital_performance[(
                (hospital_performance.index == stroke_team) & 
                (hospital_performance['stroke_type'] == stroke_type)
                )]
            # Create a fresh copy of the original performance data
            # and convert the DataFrame to a Series with squeeze():
            hospital_data_scenario = hospital_data_original.copy().squeeze(axis=0)
            hospital_data_scenario['stroke_team'] = stroke_team

            # Keep track of which scenarios are used in here:
            scenarios_list = []
            # Update the hospital data according to the scenario selected:
            if d['speed'] == 1:
                # Speed scenario            
                scenarios_list.append('speed')
                # All patients are scanned within 4hrs of arrival
                # (how does this work for the masks picking different times for MT and IVT? Someteims 4hr, sometimes 8hr) ---- check this
                for key in ['proportion3_of_mask2_with_arrival_to_scan_on_time_ivt',
                            'proportion3_of_mask2_with_arrival_to_scan_on_time_mt']:
                    hospital_data_scenario[key] = scenario_vals_dict['arrival_to_scan_on_time_proportion']
                # Update mean arrival time to be exactly the target value
                # or the current value, whichever is smaller.
                for key in ['lognorm_mu_arrival_scan_arrival_mins_ivt',
                            'lognorm_mu_arrival_scan_arrival_mins_mt']:
                    hospital_data_scenario[key] = np.minimum(
                        scenario_vals_dict['lognorm_mu_arrival_scan_arrival_mins'], 
                        hospital_data_scenario[key])
                # Update variation in arrival time.
                for key in ['lognorm_sigma_arrival_scan_arrival_mins_ivt',
                            'lognorm_sigma_arrival_scan_arrival_mins_mt']:
                    hospital_data_scenario[key] = scenario_vals_dict['lognorm_sigma_arrival_scan_arrival_mins']

            if d['onset'] == 1:
                # Onset time scenario
                scenarios_list.append('onset')
                # More patients have their onset time determined
                for key in ['proportion1_of_all_with_onset_known_ivt',
                            'proportion1_of_all_with_onset_known_mt']:
                    prop = onset_time_known_proportion_dict[stroke_type]
                    if hospital_data_scenario[key] < prop:
                        hospital_data_scenario[key] = prop

            if d['benchmark'] == 1:
                # Benchmark scenario
                scenarios_list.append('benchmark')

                mask = (
                    (df_benchmark_codes.index == stroke_team) &
                    (df_benchmark_codes['Stroke type'] == stroke_type)
                    )
                # The proportion of eligible patients receiving treatment is 
                # in line with the benchmark teams' proportions.
                hospital_data_scenario['proportion6_of_mask5_with_treated_ivt'] = (
                    df_benchmark_codes['Benchmark rate of mask5'][mask].values[0]
                )
            
            # Build the scenario name from the options selected above.
            # If none of the options are selected, name this scenario "base".
            scenario_name = ('base' if len(scenarios_list) == 0
                             else ' + '.join(scenarios_list))
            hospital_data_scenario['scenario'] = scenario_name

            hospital_data_scenario.name = f'{stroke_team} / {stroke_type} / {scenario_name}'
            
            # Recalculate treatment rates to reflect the changes.
            # n.b. if this calculation is done for "base" scenario,
            # it will return different answers than the start 
            # values. This is because in the real data, some 
            # patients do not meet all of these conditions yet
            # still receive treatment.
            rates = []
            for treatment in ['ivt', 'mt']:
                rate = 1.0
                keys = [
                    'proportion1_of_all_with_onset_known_',
                    'proportion2_of_mask1_with_onset_to_arrival_on_time_',
                    'proportion3_of_mask2_with_arrival_to_scan_on_time_',
                    'proportion4_of_mask3_with_onset_to_scan_on_time_',
                    'proportion5_of_mask4_with_enough_time_to_treat_',
                    'proportion6_of_mask5_with_treated_'
                    ]
                for key in keys:
                    rate *= hospital_data_scenario[key + treatment] 
                rates.append(rate)
            hospital_data_scenario['proportion_of_all_with_mask6_and_ivt'] = rates[0]
            hospital_data_scenario['proportion_of_all_with_mask6_and_mt'] = rates[1]
            # Leave proportion of MT with IVT as it is.
            if scenario_name != 'base':
                hospital_data_scenario['proportion_of_all_with_ivt'] = rates[0]
                hospital_data_scenario['proportion_of_all_with_mt'] = rates[1]
            
            # Store this data in the results dataframe.
            if count == 0:
                df_performance_scenarios = hospital_data_scenario.copy()
            else:
                # Combine the two Series into a single DataFrame:
                df_performance_scenarios = pd.merge(
                    df_performance_scenarios, hospital_data_scenario,
                    right_index=True, left_index=True)
            count += 1

    # Transpose the dataframe to match the original hospital_performance 
    # format:
    return df_performance_scenarios.T


def set_up_results_dataframe():
    """
    Set up the names of the results to record from the trials.
    
    Make sure that what happens in here lines up with the contents of:
    + gather_summary_results_across_all_trials()
    + gather_results_from_trial()
    """
    # Record these measures...
    outcome_results_columns = [
        'Percent_Thrombolysis',
        # 'Percent_Thrombolysis',
        'Baseline_good_outcomes_per_1000_patients',
        'Additional_good_outcomes_per_1000_patients'
        # 'onset_to_needle_mins',
        # 'onset_to_puncture_mins'
    ]

    # ... with these stats...
    results_types = [
        '_(median)',
        '_(low_5%)',
        '_(high_95%)',
        '_(mean)',
        '_(stdev)',
        '_(95ci)',
    ]
    # ... and gather all combinations of measure and stat here:
    results_columns = [column + ending for column in outcome_results_columns
                       for ending in results_types]

    # Also store onset to needle time:
    # results_columns += ['Onset_to_needle_(mean)']
    results_columns += ['Onset_to_needle_(mean)']
    results_columns += ['Onset_to_puncture_(mean)']
    results_columns += ['stroke_team']
    results_columns += ['scenario']

    results_df = pd.DataFrame(columns=results_columns)

    # trial dataframe is set up each scenario, but define column names here
    trial_df_columns = outcome_results_columns + results_columns[-4:-2]
    
    return results_df, outcome_results_columns, trial_df_columns


def gather_summary_results_across_all_trials(outcome_results_columns, trial_df):
    """
    Gather results across all trials.

    Inputs:
    -------
    outcome_results_columns - list. Names of things measured from the
                              outcome model results. Output from
                              set_up_results_dataframe().
    trial_df                - pd.DataFrame. A dataframe for the results
                              to go into. Has as many rows as there
                              have already been trials.

    Returns:
    --------
    summary_trial_results - list. Important results gathered from the
                            trial dataframe.
    """
    
    number_of_trials = len(trial_df.index)
    
    summary_trial_results = []

    # stick a bunch of if/elif in here to maintain the input order?
    
    # sometimes these medians etc. are calculated when there's only one or two valid values in the column. Should probably do something about that.
    for column in outcome_results_columns:
        
        scale = (trial_df[column].std() / sqrt(number_of_trials) 
                 if number_of_trials > 0 else np.NaN)
        ci95 = (trial_df[column].mean() -
                stats.norm.interval(0.95, loc=trial_df[column].mean(),
                scale=scale)[0]
               if ((np.isnan(scale) == False) & (scale != 0.0)) else np.NaN)
        
        results_here = [
            trial_df[column].median(),
            trial_df[column].quantile(0.05),
            trial_df[column].quantile(0.95),
            trial_df[column].mean(),
            trial_df[column].std(),
            ci95,
        ]
        summary_trial_results += results_here
    summary_trial_results += [
        trial_df['Onset_to_needle_(mean)'].mean()
        if np.all(np.isnan(trial_df['Onset_to_needle_(mean)'])) == False
        else np.NaN]
    summary_trial_results += [
        trial_df['Onset_to_puncture_(mean)'].mean()
        if np.all(np.isnan(trial_df['Onset_to_puncture_(mean)'])) == False
        else np.NaN]
    
    return summary_trial_results


def gather_results_from_trial(
        trial_columns, combo_trial_dict, results_by_stroke_type, 
        n_baseline_good_per_1000, n_additional_good_per_1000
        ):
    """
    Gather results for this trial only.
    
    This setup is a bit silly but allows for trial_columns being in any order.

    Inputs:
    -------
    trial_columns              - list. Names of columns to pick out
                                 results for here. Output from 
                                 set_up_results_dataframe().
    combo_trial_dict           - dict. Results of pathway simulations.
                                 Output of run_trial_of_pathways().
    results_by_stroke_type     - dict. Results of outcome model.
                                 Output of run_discrete_outcome_model.
    n_baseline_good_per_1000   - float. Number of good outcomes per 
                                 1000 patients when nobody is treated.
    n_additional_good_per_1000 - float. Difference in number of good
                                 outcomes per 1000 patients between
                                 this scenario and the baseline.

    Returns:
    --------
    result - list. The useful information.
    """

    # Save scenario results to dataframe
    result = []
    for key in trial_columns:
        if key == 'Percent_Thrombolysis':
            vals = np.mean(combo_trial_dict['ivt_chosen_bool'])*100.0
        elif key == 'Percent_Thrombolysis':
            vals = np.mean(combo_trial_dict['mt_chosen_bool'])*100.0
        elif key == 'Baseline_good_outcomes_per_1000_patients':
            vals = n_baseline_good_per_1000
        elif key == 'Additional_good_outcomes_per_1000_patients':
            vals = n_additional_good_per_1000
        elif key == 'Onset_to_needle_(mean)':
            # Mean treatment times:
            # (if/else to prevent mean of empty slice RunTime warning)
            # Treatment times are only not NaN when the patients 
            # received treatment.
            needle_arr = combo_trial_dict['onset_to_needle_mins'][
                combo_trial_dict['ivt_chosen_bool'] == True]
            onset_to_needle_mins_mean = (
                np.nanmean(needle_arr) 
                if np.all(np.isnan(needle_arr)) == False
                else np.NaN)
            vals = onset_to_needle_mins_mean
        elif key == 'Onset_to_puncture_(mean)':    
            # Mean treatment times:
            # (if/else to prevent mean of empty slice RunTime warning)
            # Treatment times are only not NaN when the patients 
            # received treatment.            
            puncture_arr = combo_trial_dict['onset_to_puncture_mins'][
                combo_trial_dict['mt_chosen_bool'] == True]
            onset_to_puncture_mins_mean = (
                np.nanmean(puncture_arr) 
                if np.all(np.isnan(puncture_arr)) == False
                else np.NaN)
            vals = onset_to_puncture_mins_mean
        else:
            raise KeyError(f'Missing key: {key}')
    
        result.append(vals)
        
    return result


def set_up_pathway_objects(
        hospital_name='', lvo_data=None, nlvo_data=None, other_data=None
        ):
    """
    Create pathway simulation objects for eligible stroke types.

    Inputs:
    -------
    hospital_name - str. Stroke team name for naming the objects.
    lvo_data      - pd.Series. The hospital performance parameters
                    for LVO patients.
    nlvo_data     - pd.Series. The hospital performance parameters
                    for nLVO patients.
    other_data    - pd.Series. The hospital performance parameters
                    for "other" patients.

    Returns:
    --------
    pathway_object_dict - dict. Contains the pathway objects.
    """
    pathway_object_dict = {}
    if lvo_data is not None:
        patient_pathway_lvo = SSNAP_Pathway(
            hospital_name, lvo_data, stroke_type_code=2)
        pathway_object_dict['lvo'] = patient_pathway_lvo
    if nlvo_data is not None:
        patient_pathway_nlvo = SSNAP_Pathway(
            hospital_name, nlvo_data, stroke_type_code=1)
        pathway_object_dict['nlvo'] = patient_pathway_nlvo
    if other_data is not None:
        patient_pathway_other = SSNAP_Pathway(
            hospital_name, other_data, stroke_type_code=0)
        pathway_object_dict['other'] = patient_pathway_other
    return pathway_object_dict


def run_trial_of_pathways(pathway_object_dict):
    """
    Run the pathway simulations for multiple stroke types.

    The pathways are run separately and with different hospital
    performance parameters for the different stroke types.
    The results from the multiple pathways are then combined into
    one single output dictionary.

    Inputs:
    -------
    pathway_object_dict - dict. The hospital performance parameters
                          for the separate stroke types.

    Returns:
    --------
    combo_trial_dict - dict. The combined results from the multiple
                       pathway simulations.
    """
    
    # Gather the results in here:
    trial_dict_list = []
    
    # LVO
    try:
        patient_pathway_lvo = pathway_object_dict['lvo']
        patient_pathway_lvo.run_trial()
        trial_dict_list.append(patient_pathway_lvo.trial)
    except KeyError:
        pass  # Don't add to the trial results list.
    
    # nLVO
    try:
        patient_pathway_nlvo = pathway_object_dict['nlvo']
        patient_pathway_nlvo.run_trial()
        trial_dict_list.append(patient_pathway_nlvo.trial)
    except KeyError:
        pass  # Don't add to the trial results list.
    
    # Other
    try:
        patient_pathway_other = pathway_object_dict['other']
        patient_pathway_other.run_trial()
        trial_dict_list.append(patient_pathway_other.trial)
    except KeyError:
        pass  # Don't add to the trial results list.

    # Combine results of the three stroke types:
    # The empty list is in concatenate() in case there's only one entry
    # in trial_dict_list.
    combo_trial_arrays = [
        np.concatenate(
            ([np.array([], dtype=trial_dict_list[0][key].data.dtype)] + 
             [d[key].data for d in trial_dict_list]), 
            dtype=trial_dict_list[0][key].data.dtype
            ) 
        for key in patient_pathway_lvo.trial.keys()
        ]
    combo_trial_dict = dict(zip(trial_dict_list[0].keys(), combo_trial_arrays))

    return combo_trial_dict


def run_discrete_outcome_model(patient_pathway_dict):
    """
    Wrapper to run the discrete outcome model.

    The discrete outcome model needs a series of pre-stroke mRS scores,
    one for each patient in the data. This function assumes that that
    data is unavailable and so invents an mRS score via the "x_scores"
    array. The x-scores are sampled uniformly in probability from 0 to 
    1 and the values are matched to a pre-stroke mRS distribution to
    select the mRS bin that each x falls into.

    Inputs:
    -------
    patient_pathway_dict - dict. Data for the discrete outcome model.
                           Named so because it's assumed this will be
                           the output from a patient pathway.

    Returns:
    --------
    results_by_stroke_type - dict. Outcome results split by stroke type
                             and treatment.
    patient_array_outcomes - dict. Combined outcome result for an input
                             patient data set, picking out the relevant
                             parts from the split results dictionary.
    """
    # Import required bits from the stroke-outcome package:
    from stroke_outcome.outcome_utilities import \
        import_mrs_dists_from_file, import_utility_dists_from_file
    from stroke_outcome.discrete_outcome import Discrete_outcome
    
    mrs_dists, mrs_dists_notes = import_mrs_dists_from_file()
    utility_dists, utility_dists_notes = (
        import_utility_dists_from_file())

    number_of_patients = len(patient_pathway_dict['stroke_type_code']) 

    # Initiate the outcome model object:
    discrete_outcome = Discrete_outcome(
        mrs_dists,
        number_of_patients,
        utility_dists.loc['Wang2020'].values  # Default utility scores.
        )
    # Import patient array data:
    for key in discrete_outcome.trial.keys():
        if key in patient_pathway_dict:
            discrete_outcome.trial[key].data = patient_pathway_dict[key]

    # Invent some "x" scores:
    x_scores = np.random.uniform(low=0.0, high=1.0, size=number_of_patients)
    discrete_outcome.trial['x_pre_stroke'].data = x_scores

    # Calculate outcomes:
    import copy
    results_by_stroke_type, patient_array_outcomes = discrete_outcome.calculate_outcomes()
    return copy.copy(results_by_stroke_type), copy.copy(patient_array_outcomes)
