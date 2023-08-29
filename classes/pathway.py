import numpy as np
import pandas as pd

from classes.evaluated_array import Evaluated_array

class SSNAP_Pathway:
    """
    Model of stroke pathway.

    This model simulates the passage through the emergency stroke 
    pathway for a cohort of patients. Each patient spends different
    amounts of time in each step of the pathway and may or may not
    meet various conditions for treatment. The calculations
    for all patients are performed simultaneously.
    
    ----- Method summary -----
    Patient times through the pathway are sampled from distributions 
    passed to the model using NumPy. Then any Yes or No choice is 
    guided by target hospital performance data, so for example if the
    target proportion of known onset times is 40%, we randomly pick 
    40% of these patients to have known onset times.
    
    The goal is to find out whether each patient passed through the
    pathway on time for treatment and then whether they are selected
    for treatment. There are separate time limits for thrombolysis
    and thrombectomy.
    
    The resulting arrays are then sanity checked against more 
    proportions from the target hospital performance data.
    A series of masks are created here with conditions matching those
    used to extract the hospital performance data, so that these masks
    can be used to calculate the equivalent metric for comparison of 
    the generated and target data.
    
    ----- Inputs -----
    + Hospital name string
    + Hospital-specific data dictionary containing:
      - Number of admissions (for patients per run)
      - Proportions of patients in the following categories...
        - onset time known
        - known arrival within the time limit
        - arrival to scan within the time limit
        - onset to scan within the time limit
        - chosen for thrombolysis
        - chosen for thrombectomy
        ... with one set for thrombolysis time limit and a second set
        for the thrombectomy time limit.
      - Log-normal number generation parameters mu and sigma for
        each of the following:
        - onset to arrival time
        - arrival to scan time
        - scan to needle time (thrombolysis)
        - scan to puncture time (thrombectomy)
        mu is the mean and sigma the standard deviation of the
        hospital's performance in these times in log-normal space.
        Again, there is one set of these mu and sigma for the 
        thrombolysis time limit and a second set for the thrombectomy 
        limit.
        
    A method in this class checks whether the hospital data dictionary
    contains all of the keys that this class expects.
    
    ----- Results -----
    The main results are, for each patient:
    + Arrival time
      + Is onset time known?                               (True/False)
      + Onset to arrival time                                 (minutes)
      + Is time below the thrombolysis limit?              (True/False)
      + Is time below the thrombectomy limit?              (True/False)
    + Scan time
      + Arrival to scan time                                  (minutes)
      + Is time below the thrombolysis limit?              (True/False)
      + Is time below the thrombectomy limit?              (True/False)
      + Onset to scan time                                    (minutes)
      + Is time below the thrombolysis limit?              (True/False)
      + Is time below the thrombectomy limit?              (True/False)
    + Thrombolysis decision
      + How much time is left for thrombolysis after scan?    (minutes)
      + Is there enough time left for thrombolysis?        (True/False)
      + Is thrombolysis given?                             (True/False)
      + Scan to needle time                                   (minutes)
      + Onset to needle time                                  (minutes)
    + Thrombolysis masks
      1. Onset time is known.                              (True/False)
      2. Mask 1 and onset to arrival time below limit.     (True/False)
      3. Mask 2 and arrival to scan time below limit.      (True/False)
      4. Mask 3 and onset to scan time below limit.        (True/False)
      5. Mask 4 and enough time left for thrombolysis.     (True/False)
      6. Mask 5 and the patient received thrombolysis.     (True/False)
    + Thrombectomy decision
      + How much time is left for thrombectomy after scan?    (minutes)
      + Is there enough time left for thrombectomy?        (True/False)
      + Is thrombectomy given?                             (True/False)
      + Scan to puncture time                                 (minutes)
      + Onset to puncture time                                (minutes)
    + Thrombectomy masks
      1. Onset time is known.                              (True/False)
      2. Mask 1 and onset to arrival time below limit.     (True/False)
      3. Mask 2 and arrival to scan time below limit.      (True/False)
      4. Mask 3 and onset to scan time below limit.        (True/False)
      5. Mask 4 and enough time left for thrombectomy.     (True/False)
      6. Mask 5 and the patient received thrombectomy.     (True/False)
    + Stroke type code                         (0=Other, 1=nLVO, 2=LVO)
    
    ----- Code layout -----
    Each of the above results is stored as a numpy array containing one
    value for each patient. Any Patient X appears at the same position 
    in all arrays, so the arrays can be seen as columns of a 2D table 
    of patient data. The main run_trial() function outputs all of the
    arrays as a single pandas DataFrame that can be saved to csv. The
    individual arrays are stored in the self.trial dictionary 
    attribute and can be accessed with the following syntax:
      Evaluated_array.trial['onset_to_arrival_mins'].data
    
    These acronyms are used to prevent enormous variable names:
    + IVT = intra-veneous thrombolysis
    + MT = mechanical thrombectomy
    + LVO = large-vessel occlusion
    + nLVO = non-large-vessel occlusion
    
    "Needle" refers to thrombolysis and "puncture" to thrombectomy.
    "On time" means within the time limit, e.g. 4hr for thrombolysis.
    """
    # #########################
    # ######### SETUP #########
    # #########################
    # Settings that are common across all stroke teams and all trials:



    # Assume these time limits for the checks at each point
    # (e.g. is onset to arrival time below 4 hours?)
    limit_ivt_mins = 4*60
    limit_mt_mins = 6*60  # ------------------------------------------------------- need to check for a reasonable number here

    # Generic time limits in minutes:
    # - allowed onset to needle time (thrombolysis)
    # - allowed overrun for slow scan to needle (thrombolysis)
    # - allowed onset to puncture time (thrombectomy)
    # - allowed overrun for slow scan to puncture (thrombectomy)
    allowed_onset_to_needle_time_mins = 270  # 4h 30m
    allowed_overrun_for_slow_scan_to_needle_mins = 15
    allowed_onset_to_puncture_time_mins = 8*60  # --------------------------------- need to check for a reasonable number here
    allowed_overrun_for_slow_scan_to_puncture_mins = 15

    
    def __init__(self, hospital_name: str, target_data_dict: dict, 
                 stroke_type_code: int or array = 0,
                 time_for_transfer: float=0.0,
                # proportion_lvo=0.35,
                # proportion_nlvo=0.65
                ):
        """
        Sets up the data required to calculate the patient pathways.

        Inputs:
        -------
        hospital_name     - str. Label for this hospital. Only used for
                            printing information about the object.
        target_data_dict  - dictionary or pandas Series. See
                            self._run_sanity_check_on_hospital_data() 
                            for the expected contents of this.
        time_for_transfer - float. Optional additional time to add on
                            to onset to puncture time to account for
                            transfer between hospitals.

        Initialises:
        ------------
        + A copy of the input target data dictionary.
        
        + Results dictionary named "trial" with the following keys:
          - onset_time_known_bool
          - onset_to_arrival_mins
          - onset_to_arrival_on_time_ivt_bool 
          - onset_to_arrival_on_time_mt_bool
          - arrival_to_scan_mins 
          - arrival_to_scan_on_time_ivt_bool
          - arrival_to_scan_on_time_mt_bool 
          - onset_to_scan_mins
          - onset_to_scan_on_time_ivt_bool 
          - time_left_for_ivt_after_scan_mins
          - enough_time_for_ivt_bool 
          - ivt_chosen_bool 
          - scan_to_needle_mins
          - onset_to_needle_mins 
          - ivt_mask1_onset_known
          - ivt_mask2_mask1_and_onset_to_arrival_on_time
          - ivt_mask3_mask2_and_arrival_to_scan_on_time
          - ivt_mask4_mask3_and_onset_to_scan_on_time
          - ivt_mask5_mask4_and_enough_time_to_treat
          - ivt_mask6_mask5_and_treated 
          - onset_to_scan_on_time_mt_bool
          - time_left_for_mt_after_scan_mins 
          - enough_time_for_mt_bool
          - mt_chosen_bool 
          - scan_to_puncture_mins 
          - onset_to_puncture_mins
          - mt_mask1_onset_known 
          - mt_mask2_mask1_and_onset_to_arrival_on_time
          - mt_mask3_mask2_and_arrival_to_scan_on_time
          - mt_mask4_mask3_and_onset_to_scan_on_time
          - mt_mask5_mask4_and_enough_time_to_treat 
          - mt_mask6_mask5_and_treated
          - stroke_type_code
          Access the data in the "trial" attribute with this syntax:
            Evaluated_array.trial['onset_to_arrival_mins'].data
        """
        try:
            self.hospital_name = str(hospital_name)
        except TypeError:
            print('Hospital name should be a string. Name not set.')
            self.hospital_name = ''

        # Run sanity checks.
        # If the data has problems, these raise an exception.
        self._run_sanity_check_on_hospital_data(target_data_dict)
        self.target_data_dict = target_data_dict
        # From hospital data, this is used often enough to give it its
        # own attribute:
        self.patients_per_run = int(target_data_dict['admissions'])
        
        # Store the transfer time between hospitals:
        self.time_for_transfer = time_for_transfer
        
        # Assign stroke type codes:
        self.stroke_type_code = stroke_type_code
                
        
        # # Assume these patient proportions:
        # self.proportion_lvo = proportion_lvo  # 0.35
        # self.proportion_nlvo = proportion_nlvo  # 0.65
        # # If these do not sum to 1, the remainder will be assigned to
        # # all other stroke types combined (e.g. haemorrhage).
        # # They're not subdivided more finely because the outcome model
        # # can only use nLVO and LVO at present (May 2023).
        
        
    def _create_fresh_trial_dict(self):
        """
        Set up a new dictionary ready for the trial results.
        
        Each entry in the dictionary here is a class Evaluated_array
        that provides sanity checks on any data that is attempted to
        be saved into the dict. The class checks whether the data is
        the expected data type and falls within the expected range.
        """
        # The following lines set up the arrays using the Evaluated_array
        # class, which performs basic sanity checks on the values of
        # the array when we set them later.
        # Each array is set up with a list of allowed data types 
        # (dtypes) and minimum and maximum allowed values if 
        # applicable. The syntax of the class is:
        # Evaluated_array(number_of_patients, valid_dtypes_list, 
        #               valid_min, valid_max)
        # n.b. the order of this dictionary on set-up is the same as
        # the order of the columns in the final results DataFrame.
        n = self.patients_per_run  # Defined to shorten the following.
        self.trial = dict(
            #
            # Initial steps
            onset_time_known_bool = Evaluated_array(n, ['int', 'bool'], 0, 1),
            onset_to_arrival_mins = Evaluated_array(n, ['float'], 0.0, np.inf),
            onset_to_arrival_on_time_ivt_bool = (
                Evaluated_array(n, ['int', 'bool'], 0, 1)),
            onset_to_arrival_on_time_mt_bool = (
                Evaluated_array(n, ['int', 'bool'], 0, 1)),
            arrival_to_scan_mins = Evaluated_array(n, ['float'], 0.0, np.inf),
            arrival_to_scan_on_time_ivt_bool = (
                Evaluated_array(n, ['int', 'bool'], 0, 1)),
            arrival_to_scan_on_time_mt_bool = (
                Evaluated_array(n, ['int', 'bool'], 0, 1)),
            onset_to_scan_mins = Evaluated_array(n, ['float'], 0.0, np.inf),
            #
            # IVT (thrombolysis)
            onset_to_scan_on_time_ivt_bool = (
                Evaluated_array(n, ['int', 'bool'], 0, 1)),
            time_left_for_ivt_after_scan_mins = (
                Evaluated_array(n, ['float'], 0.0, np.inf)),
            enough_time_for_ivt_bool = (
                Evaluated_array(n, ['int', 'bool'], 0, 1)),
            ivt_chosen_bool = Evaluated_array(n, ['int', 'bool'], 0, 1),
            scan_to_needle_mins = Evaluated_array(n, ['float'], 0.0, np.inf),
            onset_to_needle_mins = Evaluated_array(n, ['float'], 0.0, np.inf),
            #
            # Masks of IVT pathway to match input hospital performance
            ivt_mask1_onset_known = Evaluated_array(n, ['int', 'bool'], 0, 1),
            ivt_mask2_mask1_and_onset_to_arrival_on_time = (
                Evaluated_array(n, ['int', 'bool'], 0, 1)),
            ivt_mask3_mask2_and_arrival_to_scan_on_time = (
                Evaluated_array(n, ['int', 'bool'], 0, 1)),
            ivt_mask4_mask3_and_onset_to_scan_on_time = (
                Evaluated_array(n, ['int', 'bool'], 0, 1)),
            ivt_mask5_mask4_and_enough_time_to_treat = (
                Evaluated_array(n, ['int', 'bool'], 0, 1)), 
            ivt_mask6_mask5_and_treated = (
                Evaluated_array(n, ['int', 'bool'], 0, 1)),
            #
            # MT (thrombectomy)
            onset_to_scan_on_time_mt_bool = (
                Evaluated_array(n, ['int', 'bool'], 0, 1)),
            time_left_for_mt_after_scan_mins = (
                Evaluated_array(n, ['float'], 0.0, np.inf)),
            enough_time_for_mt_bool = (
                Evaluated_array(n, ['int', 'bool'], 0, 1)),
            mt_chosen_bool = Evaluated_array(n, ['int', 'bool'], 0, 1),
            scan_to_puncture_mins = Evaluated_array(n, ['float'], 0.0, np.inf),
            onset_to_puncture_mins = Evaluated_array(n, ['float'], 0.0, np.inf),
            # Masks of MT pathway to match input hospital performance
            mt_mask1_onset_known = Evaluated_array(n, ['int', 'bool'], 0, 1),
            mt_mask2_mask1_and_onset_to_arrival_on_time = (
                Evaluated_array(n, ['int', 'bool'], 0, 1)),
            mt_mask3_mask2_and_arrival_to_scan_on_time = (
                Evaluated_array(n, ['int', 'bool'], 0, 1)),
            mt_mask4_mask3_and_onset_to_scan_on_time = (
                Evaluated_array(n, ['int', 'bool'], 0, 1)),
            mt_mask5_mask4_and_enough_time_to_treat = (
                Evaluated_array(n, ['int', 'bool'], 0, 1)),
            mt_mask6_mask5_and_treated = (
                Evaluated_array(n, ['int', 'bool'], 0, 1)),
            #
            # Use the treatment decisions to assign stroke type
            stroke_type_code = (Evaluated_array(n, ['int'], 0, 2)),
        )
        
        # Also create a new blank dictionary of trial performance metrics:
        self.trial_performance_dict = {
            'stroke_team': self.hospital_name,
            'admissions': self.patients_per_run,
            'proportion_of_all_with_ivt': np.NaN,
            'proportion_of_all_with_mt': np.NaN,
            'proportion_of_mt_with_ivt': np.NaN,
            'proportion1_of_all_with_onset_known_ivt':  np.NaN,
            'proportion2_of_mask1_with_onset_to_arrival_on_time_ivt': np.NaN,
            'proportion3_of_mask2_with_arrival_to_scan_on_time_ivt': np.NaN,
            'proportion4_of_mask3_with_onset_to_scan_on_time_ivt': np.NaN,
            'proportion5_of_mask4_with_enough_time_to_treat_ivt': np.NaN,
            'proportion6_of_mask5_with_treated_ivt': np.NaN,
            'proportion1_of_all_with_onset_known_mt': np.NaN,
            'proportion2_of_mask1_with_onset_to_arrival_on_time_mt': np.NaN,
            'proportion3_of_mask2_with_arrival_to_scan_on_time_mt': np.NaN,
            'proportion4_of_mask3_with_onset_to_scan_on_time_mt': np.NaN,
            'proportion5_of_mask4_with_enough_time_to_treat_mt': np.NaN,
            'proportion6_of_mask5_with_treated_mt': np.NaN,
            'lognorm_mu_onset_arrival_mins_ivt': np.NaN,
            'lognorm_sigma_onset_arrival_mins_ivt': np.NaN,
            'lognorm_mu_arrival_scan_arrival_mins_ivt': np.NaN,
            'lognorm_sigma_arrival_scan_arrival_mins_ivt': np.NaN,
            'lognorm_mu_scan_needle_mins_ivt': np.NaN,
            'lognorm_sigma_scan_needle_mins_ivt': np.NaN,
            'lognorm_mu_onset_arrival_mins_mt': np.NaN,
            'lognorm_sigma_onset_arrival_mins_mt': np.NaN,
            'lognorm_mu_arrival_scan_arrival_mins_mt': np.NaN,
            'lognorm_sigma_arrival_scan_arrival_mins_mt': np.NaN,
            'lognorm_mu_scan_puncture_mins_mt': np.NaN,
            'lognorm_sigma_scan_puncture_mins_mt': np.NaN
            }


    def __str__(self):
        """Prints info when print(Instance) is called."""
        print_str = ''.join([
            f'For hospital {self.hospital_name}, the target data is: '
        ])
        for (key, val) in zip(
                self.target_data_dict.keys(),
                self.target_data_dict.values()
                ):
            print_str += '\n'
            print_str += f'  {key:50s} '
            print_str += f'{repr(val)}'
        
        print_str += '\n\n'
        print_str += ''.join([
            'The main useful attribute is self.trial, ',
            'a dictionary of the results of the trial.'
            ])
        print_str += ''.join([
            '\n',
            'The easiest way to create the results is:\n',
            '  Evaluated_array.run_trial()\n',    
            'Access the data in the trial dictionary with this syntax:\n'
            '  Evaluated_array.trial[\'onset_to_arrival_mins\'].data'
            ])
        return print_str
    

    def __repr__(self):
        """Prints how to reproduce this instance of the class."""
        return ''.join([
            'SSNAP_Pathway(',
            f'hospital_name={self.hospital_name}, '
            f'target_data_dict={self.target_data_dict})'
            ])    
    
    # ########################
    # ######### MAIN #########
    # ########################
    def run_trial(self, patients_per_run: int=-1):
        """
        Create the pathway details for each patient in the trial.
        
        The pathway timings and proportions of patients meeting various
        criteria are chosen to match the target hospital data,
        for example the distributions of onset to arrival times
        and the proportion of patients with known onset time.
        
        Each of the created patient arrays contains n values, one for
        each of the n patients in the trial. The xth value in all
        lists refers to the same patient. The data arrays are stored
        in the dictionary self.trial and are outputted here as a single
        pandas DataFrame.
         
        ----- Method: ----- 
        1. Generate whether onset time is known.
           Use the target proportion of known onset time.
           Randomly select patients to have known onset time so that
           the proportion known is the same.
        
        2. Generate pathway times. 
                  <-σ--->                   
                 ^  ▃                    Use the target mu and sigma to
                 |  █▄                   make a lognorm distribution.
                 | ▆██▄                  
         Number  | ████                  Pick a time for each patient
           of    | ████▆                 from this distribution.
        patients | ██████▅▂              
                 | ████████▇▆▄▂          Check that the proportions of
                 |▆█████████████▇▆▅▄▃▂▁  patients with times under the
                 +--|----------------->  limit for IVT and MT match the
                    μ    Time            target proportions.
           This is used to create:
           + onset to arrival time where known
           + arrival to scan time
           
        3. Generate whether patients receive treatments.
           Use the target proportions of patients receiving
           thrombolysis and thrombectomy given that they meet all of:
           + known onset time
           + onset to arrival within time limit
           + arrival to scan within time limit
           + onset to scan within time limit
           + enough time left for treatments
           
           Randomly select patients that meet these conditions to
           receive thrombolysis so that the proportion matches the 
           target. Then more carefully randomly select patients to
           receive thrombectomy so that both the proportion receiving
           thrombectomy and the proportion receiving both treatments
           match the targets.
           
        4. Generate treatment times.
           Similarly to Step 2, create:
           + scan to needle time (thrombolysis)
           + scan to puncture time (thrombectomy)
           
        5. Assign stroke types.
           n.b. this is not used elsewhere in this class but is useful
                for future modelling, e.g. in the outcome modelling.
           Use target proportions of patients with each stroke type.
           Assign nLVO, LVO, and "other" stroke types to the patients
           such that the treatments given make sense. Only patients
           with LVOs may receive thrombectomy, and only patients with
           nLVO or LVO may receive thrombolysis.
        
        ----- Outputs: -----
        results_dataframe - pandas DataFrame. Contains all of the
                            useful patient array data that was created
                            during the trial run.
        
        The useful patient array data is also available in the 
        self.trial attribute, which is a dictionary.
        """
        
        # The main results will go in the "trial" dictionary.
        # Each result is an array with one value per patient.
        self._create_fresh_trial_dict()
        
        if patients_per_run > 0:
            # Overwrite the input value from the hospital data.
            self.patients_per_run = patients_per_run
        elif patients_per_run == 0:
            # Don't bother running the trial.
            # Create a dictionary of trial performance metrics to match
            # the input hospital performance data dictionary.
            self._create_trial_performance_dict()
            # And create a Dataframe of the target vs trial performance:
            self._create_performance_dataframe()
            # Place all useful outputs into a pandas Dataframe:
            results_dataframe = self._gather_results_in_dataframe()
            return results_dataframe
        else:
            pass  # Don't update anything.
        
        
        # Assign randomly whether the onset time is known
        # in the same proportion as the real performance data.
        self._generate_onset_time_known_binomial()
        self._create_masks_onset_time_known()
        # Generate pathway times for all patients.
        # These sets of times are all independent of each other.
        self._sample_onset_to_arrival_time_lognorm()
        self._create_masks_onset_to_arrival_on_time()
        
        self._sample_arrival_to_scan_time_lognorm()
        self._create_masks_arrival_to_scan_on_time()
        
        # Combine these generated times into other measures:
        self._calculate_onset_to_scan_time()
        self._create_masks_onset_to_scan_on_time()
        
        # Is there enough time left for treatment?
        self._calculate_time_left_for_ivt_after_scan()
        self._calculate_time_left_for_mt_after_scan()
        self._create_masks_enough_time_to_treat()
        
        # Generate treatment decision
        self._generate_whether_ivt_chosen_binomial()
        self._generate_whether_mt_chosen_binomial()
        self._create_masks_treatment_given()
        
        # Generate treatment time
        self._sample_scan_to_needle_time_lognorm()
        self._sample_scan_to_puncture_time_lognorm()
        self._calculate_onset_to_needle_time()
        self._calculate_onset_to_puncture_time()

        # Create a dictionary of trial performance metrics to match
        # the input hospital performance data dictionary.
        self._create_trial_performance_dict()
        # And create a Dataframe of the target vs trial performance:
        self._create_performance_dataframe()
        # Check that proportion of patients answering "yes" to each
        # mask matches the target proportions.        
        self._sanity_check_masked_patient_proportions()
        
        # # # Assign which type of stroke it is *after* choosing which
        # # # treatments are given.
        # # self._assign_stroke_type_code()
        # # TEMPORTATTRRY
        # trial_stroke_type_code = 2 * np.random.binomial(1, 0.65, self.patients_per_run)
        if type(self.stroke_type_code) in [float, int]:
            trial_stroke_type_code = np.full(self.patients_per_run, self.stroke_type_code, dtype=int)
            self.trial['stroke_type_code'].data = trial_stroke_type_code
        else:
            self.trial['stroke_type_code'].data = self.stroke_type_code

        # Place all useful outputs into a pandas Dataframe:
        results_dataframe = self._gather_results_in_dataframe()
        
        return results_dataframe


    # ###################################
    # ##### PATHWAY TIME GENERATION #####
    # ###################################

    def _generate_lognorm_times(
            self,
            proportion_on_time: float=None,
            number_of_patients: int=None,
            mu_mt: float=None,
            sigma_mt: float=None,
            mu_ivt: float=None,
            sigma_ivt: float=None,
            label_for_printing: str=''
            ):
        """
        Generate times from a lognorm distribution and sanity check.
        
                  <-σ--->                   
                 ^  ▃                    Use the target mu and sigma to
                 |  █▄                   make a lognorm distribution.
                 | ▆██▄                  
         Number  | ████                  Pick a time for each patient
           of    | ████▆                 from this distribution.
        patients | ██████▅▂              
                 | ████████▇▆▄▂          Check that the proportions of
                 |▆█████████████▇▆▅▄▃▂▁  patients with times under the
                 +--|----------------->  limit for IVT and MT match the
                    μ    Time            target proportions.
        
        If mu and sigma for thrombectomy are provided, the generated
        times may be used for two different checks:
        - ensure proportion under thrombectomy limit matches target,
          then cut off at the thrombolysis limit and
          compare cut-off distribution's mu and sigma with targets.
        - cut off at the thrombolysis limit and
          compare cut-off distribution's mu and sigma with targets.
          
        If the distribution for thrombectomy matches the target,
        then expect cutting off this distribution at the thrombolysis
        limit to match that target too. In the real data, the two are
        identical distributions just with different cut-off points.
        
        If only mu and sigma for thombolysis are provided, the checks
        against the thrombectomy limit are not done.
                    
        ----- Inputs: -----
        proportion_on_time - float or None. The target proportion of 
                             patients with times below the limit. If
                             this is None, the checks are not made.
        number_of_patients - int. Number of times to generate. If this
                             is less than 30, the checks are not made.
        mu_mt              - float or None. Lognorm mu for times 
                             below the thrombectomy limit.
        sigma_mt           - float or None. Lognorm sigma for 
                             times below the thrombectomy limit.
        mu_ivt             - float or None. Lognorm mu for times 
                             below the thrombolysis limit.
        sigma_ivt          - float or None. Lognorm sigma for 
                             times below the thrombolysis limit.
        label_for_printing - str. Identifier for the warning
                             string printed if sanity checks fail.
                                  
        ----- Returns: -----
        times_mins - np.array. The sanity-checked generated times.
        """
        if number_of_patients is None:
            number_of_patients = self.patients_per_run
            
        # Select which mu and sigma to use:
        mu = mu_mt if mu_mt is not None else mu_ivt
        sigma = sigma_mt if sigma_mt is not None else sigma_ivt
        time_limit_mins = (self.limit_mt_mins if mu_mt is not None 
                           else self.limit_ivt_mins)
        
        # Generate times:
        times_mins = np.random.lognormal(
            mu,                  # mean
            sigma,               # standard deviation
            number_of_patients   # number of samples
            )
        # Round times to nearest minute:
        times_mins = np.round(times_mins, 0)
        # Set times below zero to zero:
        times_mins = np.maximum(times_mins, 0)

        if proportion_on_time is not None:
            times_mins = self._fudge_patients_after_time_limit(
                times_mins,
                proportion_on_time,
                time_limit_mins
                )


        # Sanity checks:
        if mu_mt is not None and sigma_mt is not None:
            mu_mt_generated, sigma_mt_generated = (
                self._calculate_lognorm_parameters(
                    times_mins[times_mins <= self.limit_mt_mins]))
            self._sanity_check_distribution_statistics(
                times_mins[times_mins <= self.limit_mt_mins],
                mu_mt,
                sigma_mt,
                mu_mt_generated,
                sigma_mt_generated,
                label=label_for_printing + ' on time for thrombectomy'
                )
        if mu_ivt is not None and sigma_ivt is not None:
            mu_ivt_generated, sigma_ivt_generated = (
                self._calculate_lognorm_parameters(
                    times_mins[times_mins <= self.limit_ivt_mins]))
            self._sanity_check_distribution_statistics(
                times_mins[times_mins <= self.limit_ivt_mins],
                mu_ivt,
                sigma_ivt,
                mu_ivt_generated,
                sigma_ivt_generated,
                label=label_for_printing + ' on time for thrombolysis'
                )

        return times_mins


    def _calculate_lognorm_parameters(self, patient_times: np.ndarray):
        """
        Calculate the lognorm mu and sigma for these times.
                
        Inputs:
        -------
        patient_times - np.ndarray. The distribution to check.
        """
        if len(patient_times) < 1:
            return np.NaN, np.NaN
        # Set all zero or negative values to 1 minute here
        # to prevent RuntimeWarning about division by zero
        # encountered in log. Choosing 1 minute instead of <1 minute
        # to prevent the mean of the log being skewed towards very
        # large negative values.
        patient_times = np.clip(patient_times, a_min=1, a_max=None)
        
        # Generated distribution statistics.
        mu_generated = np.mean(np.log(patient_times))
        sigma_generated = np.std(np.log(patient_times))
        
        return mu_generated, sigma_generated
    
    
    def _sample_onset_to_arrival_time_lognorm(self):
        """
        Assign onset to arrival time (natural log normal distribution).

        Creates:
        --------
        onset_to_arrival_mins -
            Onset to arrival times in minutes from the log-normal
            distribution. One time per patient.
        onset_to_arrival_on_time_ivt_bool -
            True or False for each patient arriving under
            the time limit for thrombolysis.
        onset_to_arrival_on_time_mt_bool -
            True or False for each patient arriving under
            the time limit for thrombectomy.
            
        Uses:
        -----
        onset_time_known_bool -
            Whether each patient has a known onset time. Created in
            _generate_onset_time_known_binomial().
        """
        # Initial array with all zero times:
        trial_onset_to_arrival_mins = np.zeros(self.patients_per_run)
        
        # Update onset-to-arrival times so that when onset time is 
        # unknown, the values are set to NaN (time).
        inds = (self.trial['onset_time_known_bool'].data == 0)
        trial_onset_to_arrival_mins[inds] = np.NaN
        
        # Find which patients have known onset times:
        inds_valid_times = np.where(trial_onset_to_arrival_mins == 0)[0]
        
        # Invent new times for this known-onset-time subgroup:
        valid_onset_to_arrival_mins = self._generate_lognorm_times(
            self.target_data_dict[
                'proportion2_of_mask1_with_onset_to_arrival_on_time_mt'],
            len(inds_valid_times),
            self.target_data_dict['lognorm_mu_onset_arrival_mins_mt'],
            self.target_data_dict['lognorm_sigma_onset_arrival_mins_mt'],
            self.target_data_dict['lognorm_mu_onset_arrival_mins_ivt'],
            self.target_data_dict['lognorm_sigma_onset_arrival_mins_ivt'],
            'onset to arrival'
            )
        # Place these times into the full patient list:
        trial_onset_to_arrival_mins[inds_valid_times] = \
            valid_onset_to_arrival_mins
        
        # Store the generated times:
        self.trial['onset_to_arrival_mins'].data = trial_onset_to_arrival_mins

        # Also generate and store boolean arrays of
        # whether the times are below given limits.
        # (NaN times return False in these bool lists.)
        self.trial['onset_to_arrival_on_time_ivt_bool'].data = (
            trial_onset_to_arrival_mins <= self.limit_ivt_mins)
        self.trial['onset_to_arrival_on_time_mt_bool'].data = (
            trial_onset_to_arrival_mins <= self.limit_mt_mins)


    def _sample_arrival_to_scan_time_lognorm(self):
        """
        Assign arrival to scan time (natural log normal distribution).
        
        Creates:
        --------
        arrival_to_scan_mins -
            Arrival to scan times in minutes from the log-normal
            distribution. One time per patient.
        arrival_to_scan_on_time_ivt_bool -
            True or False for each patient arriving under
            the time limit for thrombolysis.
        arrival_to_scan_on_time_mt_bool -
            True or False for each patient arriving under
            the time limit for thrombectomy.
            
        Uses:
        -----
        ivt_mask2_mask1_and_onset_to_arrival_on_time - 
            Mask of whether the onset to arrival time is below the
            thrombolysis limit and whether mask 1 is True 
            for each patient. Created in
            _create_masks_onset_to_arrival_on_time().
        mt_mask2_mask1_and_onset_to_arrival_on_time -
            Mask of whether the onset to arrival time is below the
            thrombectomy limit and whether mask 1 is True 
            for each patient. Created in
            _create_masks_onset_to_arrival_on_time().
        """
        # Store the generated times in here:
        trial_arrival_to_scan_mins = np.zeros(self.patients_per_run, 
                                              dtype=float)
        # Generate the times in batches to ensure that the 
        # distribution statistics match the target values in the 
        # various subgroups.
        try:
            # If these masks exist...
            mask1 = self.trial[
                'ivt_mask2_mask1_and_onset_to_arrival_on_time'].data
            mask2 = self.trial[
                'mt_mask2_mask1_and_onset_to_arrival_on_time'].data
            # ... sum them so that patients on time for IVT have values
            # of 2, patients on time for MT but not IVT have values of
            # 1, and patients not on time for either MT or IVT have 0.
            mask = np.sum([mask1, mask2], axis=0)
        except KeyError:
            # If the masks don't exist, generate times for all 
            # patients in one batch.
            mask = np.zeros(self.patients_per_run, dtype=float)
        
        # Generate a distribution of times for each of the possible
        # values in the mask:
        for b in [0, 1, 2]:
            # Patients that fall into this subgroup are located here:
            inds = np.where(mask == b)[0]
            if b == 2:
                # If on time for IVT, don't check the times against MT:
                mu_mt = None
                sigma_mt = None
                proportion = self.target_data_dict[
                    'proportion3_of_mask2_with_arrival_to_scan_on_time_ivt'] 
            else:
                # Check distribution statistics against MT targets.
                mu_mt = self.target_data_dict[
                    'lognorm_mu_arrival_scan_arrival_mins_mt']
                sigma_mt = self.target_data_dict[
                    'lognorm_sigma_arrival_scan_arrival_mins_mt']
                proportion = self.target_data_dict[
                    'proportion3_of_mask2_with_arrival_to_scan_on_time_mt']
            # Invent new times for the patient subgroup:
            masked_arrival_to_scan_mins = self._generate_lognorm_times(
                proportion,
                len(inds),
                mu_mt,
                sigma_mt,
                self.target_data_dict[
                    'lognorm_mu_arrival_scan_arrival_mins_ivt'],
                self.target_data_dict[
                    'lognorm_sigma_arrival_scan_arrival_mins_ivt'],
                'arrival to scan'
                )
            # Update those patients' times in the array:
            trial_arrival_to_scan_mins[inds] = masked_arrival_to_scan_mins

        # Store the generated times:
        self.trial['arrival_to_scan_mins'].data = trial_arrival_to_scan_mins

        # Also generate and store boolean arrays of
        # whether the times are below given limits.
        self.trial['arrival_to_scan_on_time_ivt_bool'].data = (
            trial_arrival_to_scan_mins <= self.limit_ivt_mins)
        self.trial['arrival_to_scan_on_time_mt_bool'].data = (
            trial_arrival_to_scan_mins <= self.limit_mt_mins)


    def _sample_scan_to_needle_time_lognorm(self):
        """
        Assign scan to needle time (natural log normal distribution).
        
        Creates:
        --------
        scan_to_needle_mins -
            Scan to needle times in minutes from the log-normal
            distribution. One time per patient.
        scan_to_needle_on_time_ivt_bool -
            True or False for each patient arriving under
            the time limit for thrombolysis.
        
        Uses:
        -----
        ivt_mask6_mask5_and_treated -
            Mask of whether each patient received thrombolysis
            and whether mask 5 is True for each patient. Created in
            _create_masks_treatment_given().
        """
        # Store the generated times in here:
        trial_scan_to_needle_mins = np.zeros(self.patients_per_run, 
                                              dtype=float)
        # Generate times only for the patients who answer True to 
        # this mask:
        try:
            # If mask 6 exists, use it. Patients who received 
            # thrombolysis answer True here.
            mask = self.trial['ivt_mask6_mask5_and_treated'].data
        except KeyError:
            # If mask 6 doesn't exist, generate times for all patients.
            mask = np.full(self.patients_per_run, 1, dtype=int)
            
        # Patients who answer False to the mask are given Not A Number
        # instead of a time:
        trial_scan_to_needle_mins[np.where(mask == 0)] = np.NaN
        # Patients who answer True are at these locations in the array:
        inds = np.where(mask == 1)[0]
        # Invent new times for the patient subgroup:
        masked_scan_to_needle_mins = self._generate_lognorm_times(
            None,  # don't check proportion treated on time
            len(inds),
            mu_ivt=self.target_data_dict['lognorm_mu_scan_needle_mins_ivt'],
            sigma_ivt=self.target_data_dict[
                'lognorm_sigma_scan_needle_mins_ivt'],
            label_for_printing='scan to needle'
            )
        # Update those patients' times in the array:
        trial_scan_to_needle_mins[inds] = masked_scan_to_needle_mins

        # Store the generated times:
        self.trial['scan_to_needle_mins'].data = trial_scan_to_needle_mins


    def _sample_scan_to_puncture_time_lognorm(self):
        """
        Assign scan to puncture time (natural log normal distribution).
        
        Creates:
        --------
        scan_to_puncture_mins -
            Scan to puncture times in minutes from the log-normal
            distribution. One time per patient.
        scan_to_puncture_on_time_mt_bool -
            True or False for each patient arriving under
            the time limit for thrombectomy.
        
        Uses:
        -----
        mt_mask6_mask5_and_treated -
            Mask of whether each patient received thrombectomy
            and whether mask 5 is True for each patient. Created in
            _create_masks_treatment_given().
        """
        # Store the generated times in here:
        trial_scan_to_puncture_mins = np.zeros(self.patients_per_run, 
                                              dtype=float)
        # Generate times only for the patients who answer True to 
        # this mask:
        try:
            # If mask 6 exists, use it. Patients who received 
            # thrombectomy answer True here.
            mask = self.trial['mt_mask6_mask5_and_treated'].data
        except KeyError:
            # If mask 6 doesn't exist, generate times for all patients.
            mask = np.full(self.patients_per_run, 1, dtype=int)

        # Patients who answer False to the mask are given Not A Number
        # instead of a time:
        trial_scan_to_puncture_mins[np.where(mask == 0)[0]] = np.NaN
        # Patients who answer True are at these locations in the array:
        inds = np.where(mask == 1)[0]
        # Invent new times for the patient subgroup:
        masked_scan_to_puncture_mins = self._generate_lognorm_times(
            None,       # Don't check proportion treated on time
            len(inds),  # Number of patients
            mu_mt=self.target_data_dict['lognorm_mu_scan_puncture_mins_mt'],
            sigma_mt=self.target_data_dict[
                'lognorm_sigma_scan_puncture_mins_mt'],
            label_for_printing='scan to puncture'
            )
        # Update those patients' times in the array:
        trial_scan_to_puncture_mins[inds] = masked_scan_to_puncture_mins

        # Store the generated times:
        self.trial['scan_to_puncture_mins'].data = trial_scan_to_puncture_mins


    # #######################################
    # ##### PATHWAY BINOMIAL GENERATION #####
    # #######################################
    def _generate_onset_time_known_binomial(self):
        """
        Assign whether onset time is known for each patient.

        Creates:
        --------
        onset_time_known_bool -
            True or False for each patient having a known onset time.
        """
        self.trial['onset_time_known_bool'].data = np.random.binomial(
                1,                          # Number of trials
                self.target_data_dict[      # Probability of success
                    'proportion1_of_all_with_onset_known_ivt'],
                self.patients_per_run       # Number of samples drawn
                ) == 1                      # Convert int to bool
        # n.b. the proportion1_of_all_with_onset_known_ivt is identical
        # to proportion1_of_all_with_onset_known_mt.


    def _generate_whether_ivt_chosen_binomial(self):
        """
        Generate whether patients receive thrombolysis (IVT).
        
        Use the target proportion of patients receiving
        thrombolysis given that they meet all of:
        + known onset time
        + onset to arrival within time limit
        + arrival to scan within time limit
        + onset to scan within time limit
        + enough time left for treatments

        Randomly select patients that meet these conditions to
        receive thrombolysis so that the proportion matches the target.

        Creates:
        --------
        ivt_chosen_bool - 
            True or False for each patient receiving thrombolysis.
            
        Uses:
        -----
        ivt_mask5_mask4_and_enough_time_to_treat -
            True or False for each patient being eligible for 
            thrombolysis as above. Created in 
            _create_masks_enough_time_to_treat().
        """
        prop = self.target_data_dict['proportion6_of_mask5_with_treated_ivt']
        # Only do this step if the proportion is within the allowed
        # bounds.
        if ((prop > 1.0) | (prop < 0.0) | (np.isnan(prop))):
            # Create an array with everyone set to False...
            trial_ivt_chosen_bool = np.zeros(self.patients_per_run, dtype=int)
            # Store this in self (==1 to convert to boolean).
            self.trial['ivt_chosen_bool'].data = trial_ivt_chosen_bool == 1
            return 
        
        # Find the indices of patients that meet thrombolysis criteria:
        inds_scan_on_time = np.where(
            self.trial[
                'ivt_mask5_mask4_and_enough_time_to_treat'].data == 1)[0]
        n_scan_on_time = len(inds_scan_on_time)

        # Randomly select some of these patients to receive 
        # thrombolysis in the same proportion as the target hospital
        # performance data.
        ivt_chosen_bool = np.random.binomial(
            1,                     # Number of trials
            prop,                  # ^ Probability of success
            n_scan_on_time         # Number of samples drawn
            )
        # Create an array with everyone set to False...
        trial_ivt_chosen_bool = np.zeros(self.patients_per_run, dtype=int)
        # ... and then update the chosen indices to True:
        trial_ivt_chosen_bool[inds_scan_on_time] = ivt_chosen_bool
        
        # Store this in self (==1 to convert to boolean).
        self.trial['ivt_chosen_bool'].data = trial_ivt_chosen_bool == 1


    def _generate_whether_mt_chosen_binomial(self):
        """
        Generate whether patients receive thrombectomy (MT).
        
        Use the target proportion of patients receiving
        thrombectomy given that they meet all of:
        + known onset time
        + onset to arrival within time limit
        + arrival to scan within time limit
        + onset to scan within time limit
        + enough time left for treatments

        Randomly select patients that meet these conditions to
        receive thrombectomy so that the proportion matches the 
        target. The selection is done in two steps to account for
        some patients also receiving IVT and some not, in a known
        target proportion.

        Creates:
        --------
        mt_chosen_bool - 
            True or False for each patient receiving thrombectomy.
        
        Uses:
        -----
        mt_mask5_mask4_and_enough_time_to_treat -
            True or False for each patient being eligible for 
            thrombectomy as above. Created in 
            _create_masks_enough_time_to_treat().
        ivt_chosen_bool -
            True of False for each patient receiving thrombolysis.
            Created in _generate_whether_ivt_chosen_binomial().
        """
        prop = self.target_data_dict['proportion6_of_mask5_with_treated_mt']
        # Only do this step if the proportion is within the allowed
        # bounds.
        if ((prop > 1.0) | (prop < 0.0) | (np.isnan(prop))):
            # Create an array with everyone set to False...
            trial_mt_chosen_bool = np.zeros(self.patients_per_run, dtype=int)
            # Store this in self (==1 to convert to boolean).
            self.trial['mt_chosen_bool'].data = trial_mt_chosen_bool == 1
            return 
        
        # Find how many patients could receive thrombectomy and
        # where they are in the patient array:
        inds_eligible_for_mt = np.where(self.trial[
            'mt_mask5_mask4_and_enough_time_to_treat'].data == 1)[0]
        n_eligible_for_mt = len(inds_eligible_for_mt)
        # Use the known proportion chosen to find the number of
        # patients who will receive thrombectomy:
        n_mt = np.sum(np.random.binomial(
            1,                     # Number of trials
            self.target_data_dict['proportion6_of_mask5_with_treated_mt'],
                                   # ^ Probability of success
            n_eligible_for_mt      # Number of samples drawn
            ))

        # Use the proportion of patients who receive thrombolysis and
        # thrombectomy to create two groups now.
        # Number of patients receiving both:
        if np.isnan(self.target_data_dict['proportion_of_mt_with_ivt']
                   ) == True:
            n_mt_and_ivt = 0
        else:
            n_mt_and_ivt = np.minimum(n_mt, np.sum(np.random.binomial(
                1,    # Number of trials
                self.target_data_dict['proportion_of_mt_with_ivt'],
                      # ^ Probability of success
                n_mt  # Number of samples drawn
                )))

        # Number of patients receiving thrombectomy only:
        n_mt_not_ivt = n_mt - n_mt_and_ivt

        # Find which patients in the array may be used for each group:
        inds_eligible_for_mt_and_ivt = np.where(
            (self.trial[
                'mt_mask5_mask4_and_enough_time_to_treat'].data == 1) &
            (self.trial['ivt_chosen_bool'].data == 1)
            )[0]
        inds_eligible_for_mt_not_ivt = np.where(
            (self.trial[
                'mt_mask5_mask4_and_enough_time_to_treat'].data == 1) &
            (self.trial['ivt_chosen_bool'].data == 0)
            )[0]

        # Randomly select patients from these subgroups to be given
        # thrombectomy. If the expected number is higher than the 
        # number of patients that are eligible, all of the eligible
        # patients are chosen and no more.
        if len(inds_eligible_for_mt_and_ivt) > 0:
            n_mt_and_ivt = np.minimum(n_mt_and_ivt, 
                                      len(inds_eligible_for_mt_and_ivt))
            inds_mt_and_ivt = np.random.choice(
                inds_eligible_for_mt_and_ivt,
                size=n_mt_and_ivt,
                replace=False
                )
        else:
            inds_mt_and_ivt = []
            
        if len(inds_eligible_for_mt_not_ivt) > 0:
            n_mt_not_ivt = np.minimum(n_mt_not_ivt, 
                                      len(inds_eligible_for_mt_not_ivt))
            inds_mt_not_ivt = np.random.choice(
                inds_eligible_for_mt_not_ivt,
                size=n_mt_not_ivt,
                replace=False
                )
        else:
            inds_mt_not_ivt = []

        # Initially create the array with nobody receiving treatment...
        trial_mt_chosen_bool = np.zeros(self.patients_per_run, dtype=int)
        # ... then update the patients that we've just picked out.
        trial_mt_chosen_bool[inds_mt_and_ivt] = 1
        trial_mt_chosen_bool[inds_mt_not_ivt] = 1
        # Store in self (==1 to convert to boolean):
        self.trial['mt_chosen_bool'].data = trial_mt_chosen_bool == 1
    
    
    # ##############################
    # ##### COMBINE CONDITIONS #####
    # ##############################
    def _calculate_onset_to_scan_time(self):
        """
        Find onset to scan time and boolean arrays from existing times.

        Creates:
        --------
        onset_to_scan_mins -
            Onset to scan times in minutes from the log-normal
            distribution. One time per patient.
        onset_to_scan_on_time_ivt_bool -
            True or False for each patient being scanned under
            the time limit for thrombolysis.
        onset_to_scan_on_time_mt_bool -
            True or False for each patient being scanned under
            the time limit for thrombectomy.
        
        Uses:
        -----
        onset_to_arrival_mins -
            Onset to arrival times in minutes from the log-normal
            distribution. One time per patient. Created in
            _sample_onset_to_arrival_time_lognorm().
        arrival_to_scan_mins -
            Arrival to scan times in minutes from the log-normal
            distribution. One time per patient. Created in
            _sample_arrival_to_scan_time_lognorm().
        """
        # Calculate onset to scan by summing onset to arrival and
        # arrival to scan:
        self.trial['onset_to_scan_mins'].data = np.sum([
            self.trial['onset_to_arrival_mins'].data,
            self.trial['arrival_to_scan_mins'].data,
            ], axis=0)

        # Create boolean arrays for whether each patient's time is
        # below the limit.
        self.trial['onset_to_scan_on_time_ivt_bool'].data = (
            self.trial['onset_to_scan_mins'].data <= self.limit_ivt_mins) == 1
        self.trial['onset_to_scan_on_time_mt_bool'].data = (
            self.trial['onset_to_scan_mins'].data <= self.limit_mt_mins) == 1


    def _calculate_time_left_for_ivt_after_scan(self,
                                                minutes_left: float=15.0):
        """
        Calculate the minutes left to thrombolyse after scan.
        
        Creates:
        --------
        time_left_for_ivt_after_scan_mins -
            Time left before the allowed onset to needle time in 
            minutes. If the allowed time has passed, the time left
            is set to zero. One time per patient.
        enough_time_for_ivt_bool -
            True or False for each patient having enough time left
            for thrombolysis.
        
        Uses:
        -----
        onset_to_scan_mins -
            Onset to scan times in minutes from the log-normal
            distribution. One time per patient. Created in
            _calculate_onset_to_scan_time().
        ivt_mask4_mask3_and_onset_to_scan_on_time -
            True or False for each patient being eligible for 
            thrombolysis. Created in 
            _create_masks_enough_time_to_treat().
        """
        # Calculate the time left for IVT after the scan.
        # If the time is negative, set it to -0.0.
        # The minus distinguishes the clipped values from the ones
        # that genuinely have exactly 0 minutes left.
        self.trial['time_left_for_ivt_after_scan_mins'].data = np.maximum((
            self.allowed_onset_to_needle_time_mins -
            self.trial['onset_to_scan_mins'].data
            ), -0.0)
        # True/False for whether each patient has enough time for IVT
        # given that they have known onset time and onset to arrival,
        # arrival to scan, and onset to scan below the time limit
        # *and* there are at least 15 minutes left to treat.
        self.trial['enough_time_for_ivt_bool'].data = (
            self.trial['ivt_mask4_mask3_and_onset_to_scan_on_time'].data *
            (self.trial['time_left_for_ivt_after_scan_mins'].data
                >= minutes_left)
            ) == 1

    def _calculate_time_left_for_mt_after_scan(self,
                                               minutes_left: float=15.0):
        """
        Calculate the minutes left for thrombectomy after scan.
        
        Creates:
        --------
        time_left_for_mt_after_scan_mins -
            Time left before the allowed onset to puncture time in 
            minutes. If the allowed time has passed, the time left
            is set to zero. One time per patient.
        enough_time_for_mt_bool -
            True or False for each patient having enough time left
            for thrombectomy.
        
        Uses:
        -----
        onset_to_scan_mins -
            Onset to scan times in minutes from the log-normal
            distribution. One time per patient. Created in
            _calculate_onset_to_scan_time().
        mt_mask4_mask3_and_onset_to_scan_on_time -
            True or False for each patient being eligible for 
            thrombectomy. Created in 
            _create_masks_enough_time_to_treat().
        """
        # Calculate the time left for MT after the scan.
        # If the time is negative, set it to -0.0.
        # The minus distinguishes the clipped values from the ones
        # that genuinely have exactly 0 minutes left.
        self.trial['time_left_for_mt_after_scan_mins'].data = np.maximum((
            self.allowed_onset_to_puncture_time_mins -
            self.trial['onset_to_scan_mins'].data 
            ), -0.0)

        # True/False for whether each patient has enough time for IVT
        # given that they have known onset time and onset to arrival,
        # arrival to scan, and onset to scan below the time limit
        # *and* there are at least 15 minutes left to treat.
        self.trial['enough_time_for_mt_bool'].data = (
            self.trial['mt_mask4_mask3_and_onset_to_scan_on_time'].data *
            (self.trial['time_left_for_mt_after_scan_mins'].data
                >= minutes_left)
            ) == 1

    def _calculate_onset_to_needle_time(self):
        """
        Calculate onset to needle times from existing data.
        
        Creates:
        --------
        onset_to_needle_mins -
            Onset to needle times in minutes from summing the onset to
            scan and scan to needle times and setting a maximum value
            of clip_limit_mins. One time per patient.
            If onset time is unknown, the resulting onset to needle
            time will be NaN.
        
        Uses:
        -----
        onset_to_scan_mins -
            Onset to scan times in minutes from the log-normal
            distribution. One time per patient. Created in
            _calculate_onset_to_scan_time().
        scan_to_needle_mins -
            Scan to needle times in minutes from the log-normal
            distribution. One time per patient. Created in
            _sample_scan_to_needle_time_lognorm().
        """
        onset_to_needle_mins = (
            self.trial['onset_to_scan_mins'].data +
            self.trial['scan_to_needle_mins'].data
            )
        self.trial['onset_to_needle_mins'].data = onset_to_needle_mins
        
    
    def _calculate_onset_to_puncture_time(self):
        """
        Calculate onset to puncture times from existing data.
        
        Creates:
        --------
        onset_to_puncture_mins -
            Onset to puncture times in minutes from summing the onset
            to scan and scan to puncture times and setting a maximum 
            value of clip_limit_mins. One time per patient.
            If onset time is unknown, the resulting onset to needle
            time will be NaN.
        
        Uses:
        -----
        onset_to_scan_mins -
            Onset to scan times in minutes from the log-normal
            distribution. One time per patient. Created in
            _calculate_onset_to_scan_time().
        scan_to_puncture_mins -
            Scan to puncture times in minutes from the log-normal
            distribution. One time per patient. Created in
            _sample_scan_to_puncture_time_lognorm().
        """
        onset_to_puncture_mins = (
            self.trial['onset_to_scan_mins'].data +
            self.trial['scan_to_puncture_mins'].data +
            self.time_for_transfer  # Optional, set in __init__().
            )
        self.trial['onset_to_puncture_mins'].data = onset_to_puncture_mins

        
    # ##########################
    # ##### GATHER RESULTS #####
    # ##########################
    def _gather_results_in_dataframe(self):
        """
        Combine all results arrays into a single dataframe.
        
        The gathered arrays are all contained in the trial dictionary.
        """        
        # The following construction for "data" will work as long as
        # all arrays in trial have the same length.
        df = pd.DataFrame(
            data=np.array(
                [v.data for v in self.trial.values()], dtype=object).T,
            columns=list(self.trial.keys())
            )
        self.results_dataframe = df
        
    
    def _create_trial_performance_dict(self):
        """
        Create dictionary of performance here to match input data dict.
        
        Measure and record the same metrics that were used to create 
        the patient array details in the trials run here. The keys of the
        dictionary are:
        + stroke_team 
        + admissions 
        + proportion_of_all_with_ivt
        + proportion_of_all_with_mt 
        + proportion_of_mt_with_ivt
        + proportion1_of_all_with_onset_known_ivt
        + proportion2_of_mask1_with_onset_to_arrival_on_time_ivt
        + proportion3_of_mask2_with_arrival_to_scan_on_time_ivt
        + proportion4_of_mask3_with_onset_to_scan_on_time_ivt
        + proportion5_of_mask4_with_enough_time_to_treat_ivt
        + proportion6_of_mask5_with_treated_ivt
        + lognorm_mu_onset_arrival_mins_ivt
        + lognorm_sigma_onset_arrival_mins_ivt
        + lognorm_mu_arrival_scan_arrival_mins_ivt
        + lognorm_sigma_arrival_scan_arrival_mins_ivt
        + lognorm_mu_scan_needle_mins_ivt 
        + lognorm_sigma_scan_needle_mins_ivt
        + proportion1_of_all_with_onset_known_mt
        + proportion2_of_mask1_with_onset_to_arrival_on_time_mt
        + proportion3_of_mask2_with_arrival_to_scan_on_time_mt
        + proportion4_of_mask3_with_onset_to_scan_on_time_mt
        + proportion5_of_mask4_with_enough_time_to_treat_mt
        + proportion6_of_mask5_with_treated_mt
        + lognorm_mu_onset_arrival_mins_mt
        + lognorm_sigma_onset_arrival_mins_mt
        + lognorm_mu_arrival_scan_arrival_mins_mt
        + lognorm_sigma_arrival_scan_arrival_mins_mt
        + lognorm_mu_scan_puncture_mins_mt
        + lognorm_sigma_scan_puncture_mins_mt
        
        Returns:
        --------
        trial_performance_dict - dict. Contains metrics in this trial.
        """
        # Store the results in this dictionary:
        trial_performance_dict = dict()
        
        # Directly copied from the target performance data:
        trial_performance_dict['stroke_team'] = self.hospital_name
        trial_performance_dict['admissions'] = self.patients_per_run
        
        # Calculate treatment rates across the whole cohort:
        self._calculate_treatment_rates()
        trial_performance_dict['proportion_of_all_with_ivt'] = self.ivt_rate
        trial_performance_dict['proportion_of_all_with_mt'] = self.mt_rate
        trial_performance_dict[
            'proportion_of_mt_with_ivt'] = self.mt_with_ivt_rate
        
        proportion_dict = self._calculate_trial_proportions()
        # Transfer the proportion dict contents into the trial dict:
        for key in list(proportion_dict.keys()):
            trial_performance_dict[key] = proportion_dict[key]

        # Lognorm distribution parameters:
        lognorm_dicts = [
            dict(label = 'onset_arrival_mins_ivt',
                 times = self.trial['onset_to_arrival_mins'].data,
                 mask = self.trial[
                    'ivt_mask2_mask1_and_onset_to_arrival_on_time'].data,
                 ),
            dict(label = 'arrival_scan_arrival_mins_ivt',
                 times = self.trial['arrival_to_scan_mins'].data,
                 mask = self.trial[
                    'ivt_mask3_mask2_and_arrival_to_scan_on_time'].data,
                 ),
            dict(label = 'scan_needle_mins_ivt',
                 times = self.trial['scan_to_needle_mins'].data,
                 mask = self.trial['ivt_mask6_mask5_and_treated'].data,
                 ),        
            dict(label = 'onset_arrival_mins_mt',
                 times = self.trial['onset_to_arrival_mins'].data,
                 mask = self.trial[
                    'mt_mask2_mask1_and_onset_to_arrival_on_time'].data,
                 ),
            dict(label = 'arrival_scan_arrival_mins_mt',
                 times = self.trial['arrival_to_scan_mins'].data,
                 mask = self.trial[
                    'mt_mask3_mask2_and_arrival_to_scan_on_time'].data,
                 ),
            dict(label = 'scan_puncture_mins_mt',
                 times = self.trial['scan_to_puncture_mins'].data,
                 mask = self.trial['mt_mask6_mask5_and_treated'].data,
                 )
            ]
        for d in lognorm_dicts:
            mu_generated, sigma_generated = (
                self._calculate_lognorm_parameters(d['times'][d['mask']]))
            trial_performance_dict[
                'lognorm_mu_' + d['label']] = mu_generated
            trial_performance_dict[
                'lognorm_sigma_' + d['label']] = sigma_generated
        
        # Store the dictionary in self.
        self.trial_performance_dict = trial_performance_dict
                

    def _calculate_treatment_rates(self):
        """
        Calculate treatment rates across the whole cohort.
        
        Creates:
        --------
        ivt_rate - 
            Float. The proportion of the whole cohort that received
            thrombolysis.
        mt_rate - 
            Float. The proportion of the whole cohort that received
            thrombectomy.
        mt_with_ivt_rate - 
            Float. The proportion of "patients that received 
            thrombectomy" that also received thrombolysis.
        
        Uses:
        -----
        ivt_chosen_bool -
            True of False for each patient receiving thrombolysis.
            Created in _generate_whether_ivt_chosen_binomial().
        mt_chosen_bool -
            True of False for each patient receiving thrombectomy.
            Created in _generate_whether_mt_chosen_binomial().
        """
        if len(self.trial['ivt_chosen_bool'].data) == 0:
            # This condition is met when the number of patients is
            # zero or otherwise this function is run before the 
            # treatments are assigned.
            self.ivt_rate = np.NaN
            self.mt_rate = np.NaN
            self.mt_with_ivt_rate = np.NaN
        else:
            self.ivt_rate = self.trial['ivt_chosen_bool'].data.mean()
            self.mt_rate = self.trial['mt_chosen_bool'].data.mean()

            n_mt = len(np.where(self.trial['mt_chosen_bool'].data == 1)[0])
            n_mt_with_ivt = len(np.where(
                (self.trial['mt_chosen_bool'].data == 1) &
                (self.trial['ivt_chosen_bool'].data == 1)
                )[0])
            self.mt_with_ivt_rate = ((n_mt_with_ivt / n_mt)
                                if n_mt > 0 else np.NaN)

    
    def _calculate_trial_proportions(self):
        """
        Calculate the proportions of patients passing each mask.
        
        The proportions are stored in a dictionary and can be compared
        directly with the input target hospital performance data.
        
        Returns:
        --------
        trial_performance_dict - dict. Contains the proportions with
                                 the same keys as the input target 
                                 performance data.
        """
        # Proportions of patients at each step.
        # Use these names for the resulting proportions.
        target_proportions = [
            'proportion1_of_all_with_onset_known_',
            'proportion2_of_mask1_with_onset_to_arrival_on_time_',
            'proportion3_of_mask2_with_arrival_to_scan_on_time_',
            'proportion4_of_mask3_with_onset_to_scan_on_time_',
            'proportion5_of_mask4_with_enough_time_to_treat_',
            'proportion6_of_mask5_with_treated_'
        ]
        
        # Calculate the proportions using masks with these names
        # prepended with either "ivt" or "mt":
        mask_names = [
            '_mask1_onset_known',
            '_mask2_mask1_and_onset_to_arrival_on_time',
            '_mask3_mask2_and_arrival_to_scan_on_time',
            '_mask4_mask3_and_onset_to_scan_on_time',
            '_mask5_mask4_and_enough_time_to_treat',
            '_mask6_mask5_and_treated'
        ]
        
        proportion_dict = dict()
        for i, treatment in enumerate(['ivt', 'mt']):
            for j, proportion_name in enumerate(target_proportions):
                mask_now = self.trial[treatment + mask_names[j]].data
                if j > 0:
                    # If there's a previous mask, find it from the dict:
                    mask_before = self.trial[treatment + mask_names[j-1]].data
                else:
                    # All patients answered True in the previous step.
                    mask_before = np.full(len(mask_now), 1)

                # Create patient proportions from generated data.
                # Proportion is Yes to Mask now / Yes to Mask before.
                trial_proportion = (np.sum(mask_now) / np.sum(mask_before)
                                    if np.sum(mask_before) > 0 else np.NaN)
                # Store in the dictionary.
                proportion_dict[
                    proportion_name + treatment] = trial_proportion
        return proportion_dict

    
    def _create_performance_dataframe(self):
        """
        Place target and trial performance data into one dataframe.
        
        Creates:
        --------
        df_performance - pandas DataFrame. Contains the performance
                         data of both this trial and the input target
                         values for comparison.
        """
        try:
            # If a performance dataframe already exists, add the 
            # results of this trial to it.
            df_target = self.df_performance
            trial_number = len(df_target.columns)
        except AttributeError:
            # Create a named pandas Series of the target performance 
            # data.
            if type(self.target_data_dict) == dict:
                df_target = pd.Series(
                    data = list(self.target_data_dict.values()),
                    index = list(self.target_data_dict.keys()),
                    name = 'Target'
                    )
            else:
                # Assume it's a pandas Series already.
                df_target = self.target_data_dict
                df_target.name = 'Target'
            trial_number = 1
        
        # Convert this trial dictionary to a pandas Series.
        # Number the trial based on the current number of columns
        # in the performance dataframe.
        df_trial = pd.Series(
            data = list(self.trial_performance_dict.values()),
            index = list(self.trial_performance_dict.keys()),
            name = f'Trial_{trial_number}'
            )

        
        # Combine the two Series into a single DataFrame:
        df_performance = pd.merge(df_target, df_trial, 
                                  right_index=True, left_index=True)
        self.df_performance = df_performance
    
    
    # ##############################
    # ######## CREATE MASKS ########
    # ##############################
    """
    Make masks of patients meeting the conditions.

    Create masks for subgroups of patients as in the
    hospital performance data extraction.
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
    ▏Mask 3: Is arrival to scan within the time limit?                ▕
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
    """
    def _create_masks_onset_time_known(self):
        """
        Mask 1: Is onset time known?
        
        Although this mask looks redundant, it is provided for easier
        direct comparison with the masks creating during the hospital
        performance data extraction. The IVT mask is identical to the
        MT mask and both are an exact copy of the onset known boolean
        array.
        
        Creates:
        --------
        ivt_mask1_onset_known -
            Mask of whether onset time is known for each patient.
        mt_mask1_onset_known -
            Mask of whether onset time is known for each patient.
        
        Uses:
        -----
        onset_time_known_bool - 
            Whether onset time is known for each patient. Created in
            _generate_onset_time_known_binomial().
        """
        # Same mask for thrombolysis and thrombolysis.
        mask = np.copy(self.trial['onset_time_known_bool'].data)

        self.trial['ivt_mask1_onset_known'].data = mask
        self.trial['mt_mask1_onset_known'].data = mask


    def _create_masks_onset_to_arrival_on_time(self):
        """
        Mask 2: Is arrival within x hours?
        
        Creates:
        --------
        ivt_mask2_mask1_and_onset_to_arrival_on_time -
            Mask of whether the onset to arrival time is below the
            thrombolysis limit and whether mask 1 is True 
            for each patient.
        mt_mask2_mask1_and_onset_to_arrival_on_time -
            Mask of whether the onset to arrival time is below the
            thrombectomy limit and whether mask 1 is True 
            for each patient.
        
        Uses:
        -----
        ivt_mask1_onset_known - 
            Whether onset time is known for each patient. Created in
            _create_masks_onset_time_known().
        onset_to_arrival_on_time_ivt_bool -
            Whether onset to arrival time for each patient is under 
            the thrombolysis limit. Created in 
            _sample_onset_to_arrival_time_lognorm().
        mt_mask1_onset_known - 
            Whether onset time is known for each patient. Created in
            _create_masks_onset_time_known(). 
        onset_to_arrival_on_time_mt_bool -
            Whether onset to arrival time for each patient is under 
            the thrombectomy limit. Created in 
            _sample_onset_to_arrival_time_lognorm().
        """
        mask_ivt = (
            (self.trial['ivt_mask1_onset_known'].data == 1) &
            (self.trial['onset_to_arrival_on_time_ivt_bool'].data == 1)
            )
        mask_mt = (
            (self.trial['mt_mask1_onset_known'].data == 1) &
            (self.trial['onset_to_arrival_on_time_mt_bool'].data == 1)
            )

        self.trial[
            'ivt_mask2_mask1_and_onset_to_arrival_on_time'].data = mask_ivt
        self.trial[
            'mt_mask2_mask1_and_onset_to_arrival_on_time'].data = mask_mt

        
    def _create_masks_arrival_to_scan_on_time(self):
        """
        Mask 3: Is scan within x hours of arrival?
                
        Creates:
        --------
        ivt_mask3_mask2_and_arrival_to_scan_on_time -
            Mask of whether the arrival to scan time is below the
            thrombolysis limit and whether mask 2 is True 
            for each patient.
        mt_mask3_mask2_and_arrival_to_scan_on_time -
            Mask of whether the arrival to scan time is below the
            thrombectomy limit and whether mask 2 is True 
            for each patient.
        
        Uses:
        -----
        ivt_mask2_mask1_and_onset_to_arrival_on_time - 
            IVT mask 2. Created in
            _create_masks_onset_to_arrival_on_time().
        arrival_to_scan_on_time_ivt_bool -
            Whether arrival to scan time for each patient is under 
            the thrombolysis limit. Created in 
            _sample_arrival_to_scan_time_lognorm().
        mt_mask2_mask1_and_onset_to_arrival_on_time - 
            MT mask 2. Created in
            _create_masks_onset_to_arrival_on_time(). 
        arrival_to_scan_on_time_mt_bool -
            Whether arrival to scan time for each patient is under 
            the thrombectomy limit. Created in 
            _sample_arrival_to_scan_time_lognorm().
        """
        mask_ivt = (
            (self.trial[
                'ivt_mask2_mask1_and_onset_to_arrival_on_time'].data == 1) &
            (self.trial['arrival_to_scan_on_time_ivt_bool'].data == 1)
            )
        mask_mt = (
            (self.trial[
                'mt_mask2_mask1_and_onset_to_arrival_on_time'].data == 1) &
            (self.trial['arrival_to_scan_on_time_mt_bool'].data == 1)
            )

        self.trial[
            'ivt_mask3_mask2_and_arrival_to_scan_on_time'].data = mask_ivt
        self.trial[
            'mt_mask3_mask2_and_arrival_to_scan_on_time'].data = mask_mt

        
    def _create_masks_onset_to_scan_on_time(self):
        """
        Mask 4: Is scan within x hours of onset?
                
        Creates:
        --------
        ivt_mask4_mask3_and_onset_to_scan_on_time -
            Mask of whether the onset to scan time is below the
            thrombolysis limit and whether mask 3 is True 
            for each patient.
        mt_mask4_mask3_and_onset_to_scan_on_time -
            Mask of whether the onset to scan time is below the
            thrombectomy limit and whether mask 3 is True 
            for each patient.
        
        Uses:
        -----
        ivt_mask3_mask2_and_arrival_to_scan_on_time - 
            IVT mask 3. Created in
            _create_masks_arrival_to_scan_on_time().
        onset_to_scan_on_time_ivt_bool -
            Whether arrival to scan time for each patient is under 
            the thrombolysis limit. Created in 
            _calculate_onset_to_scan_time().
        mt_mask3_mask2_and_arrival_to_scan_on_time - 
            MT mask 3. Created in
            _create_masks_arrival_to_scan_on_time(). 
        onset_to_scan_on_time_mt_bool -
            Whether arrival to scan time for each patient is under 
            the thrombectomy limit. Created in 
            _calculate_onset_to_scan_time().        
        """
        mask_ivt = (
            (self.trial[
                'ivt_mask3_mask2_and_arrival_to_scan_on_time'].data == 1) &
            (self.trial['onset_to_scan_on_time_ivt_bool'].data == 1)
            )
        mask_mt = (
            (self.trial[
                'mt_mask3_mask2_and_arrival_to_scan_on_time'].data == 1) &
            (self.trial['onset_to_scan_on_time_mt_bool'].data == 1)
            )

        self.trial[
            'ivt_mask4_mask3_and_onset_to_scan_on_time'].data = mask_ivt
        self.trial['mt_mask4_mask3_and_onset_to_scan_on_time'].data = mask_mt

        
    def _create_masks_enough_time_to_treat(self):
        """
        Mask 5: Is there enough time left for threatment?
        
        Creates:
        --------
        ivt_mask5_mask4_and_enough_time_to_treat -
            Mask of whether there is enough time before the 
            thrombolysis limit and whether mask 4 is True 
            for each patient.
        mt_mask5_mask4_and_enough_time_to_treat -
            Mask of whether there is enough time before the 
            thrombectomy limit and whether mask 4 is True 
            for each patient.
        
        Uses:
        -----
        ivt_mask4_mask3_and_onset_to_scan_on_time - 
            IVT mask 4. Created in
            _create_masks_onset_to_scan_on_time().
        enough_time_for_ivt_bool -
            Whether there is enough time left before the thrombolysis
            limit. Created in 
            _calculate_time_left_for_ivt_after_scan().
        mt_mask4_mask3_and_onset_to_scan_on_time - 
            MT mask 4. Created in
            _create_masks_onset_to_scan_on_time(). 
        enough_time_for_mt_bool -
            Whether arrival to scan time for each patient is under 
            the thrombectomy limit. Created in 
            _calculate_time_left_for_mt_after_scan().
        """
        mask_ivt = (
            (self.trial[
                'ivt_mask4_mask3_and_onset_to_scan_on_time'].data == 1) &
            (self.trial['enough_time_for_ivt_bool'].data == 1)
            )
        mask_mt = (
            (self.trial[
                'mt_mask4_mask3_and_onset_to_scan_on_time'].data == 1) &
            (self.trial['enough_time_for_mt_bool'].data == 1)
            )

        self.trial['ivt_mask5_mask4_and_enough_time_to_treat'].data = mask_ivt
        self.trial['mt_mask5_mask4_and_enough_time_to_treat'].data = mask_mt

        
    def _create_masks_treatment_given(self):
        """
        Mask 6: Was treatment given?
                
        Creates:
        --------
        ivt_mask6_mask5_and_treated -
            Mask of whether each patient received thrombolysis
            and whether mask 5 is True for each patient.
        mt_mask6_mask5_and_treated -
            Mask of whether each patient received thrombectomy
            and whether mask 5 is True for each patient.
        
        Uses:
        -----
        ivt_mask5_mask4_and_enough_time_to_treat - 
            IVT mask 5. Created in
            _create_masks_enough_time_to_treat().
        ivt_chosen_bool -
            Whether there is enough time left before the thrombolysis
            limit. Created in 
            _generate_whether_ivt_chosen_binomial().
        mt_mask5_mask4_and_enough_time_to_treat - 
            MT mask 5. Created in _create_masks_enough_time_to_treat().
        mt_chosen_bool -
            Whether arrival to scan time for each patient is under 
            the thrombectomy limit. Created in 
            _generate_whether_mt_chosen_binomial().
        """
        mask_ivt = (
            (self.trial[
                'ivt_mask5_mask4_and_enough_time_to_treat'].data == 1) &
            (self.trial['ivt_chosen_bool'].data == 1)
            )
        mask_mt = (
            (self.trial[
                'mt_mask5_mask4_and_enough_time_to_treat'].data == 1) &
            (self.trial['mt_chosen_bool'].data == 1)
            )

        self.trial['ivt_mask6_mask5_and_treated'].data = mask_ivt
        self.trial['mt_mask6_mask5_and_treated'].data = mask_mt



    # #########################
    # ##### SANITY CHECKS #####
    # #########################
    def _run_sanity_check_on_hospital_data(self, hospital_data: dict):
        """
        Sanity check the hospital data.

        Check all of the relevant keys exist and that the data is
        of the right dtype.
        
        Inputs:
        -------
        hospital_data - a dictionary or pandas Series containing the
                        keywords in the following "keys" list. 
        """
        keys = [
            # Required for the calculations:
            'admissions',
            'proportion_of_mt_with_ivt',
            'proportion1_of_all_with_onset_known_ivt',
            'proportion6_of_mask5_with_treated_ivt',
            'proportion2_of_mask1_with_onset_to_arrival_on_time_mt',
            'proportion3_of_mask2_with_arrival_to_scan_on_time_mt',
            'proportion6_of_mask5_with_treated_mt',
            'lognorm_mu_onset_arrival_mins_ivt',
            'lognorm_sigma_onset_arrival_mins_ivt', 
            'lognorm_mu_arrival_scan_arrival_mins_ivt', 
            'lognorm_sigma_arrival_scan_arrival_mins_ivt', 
            'lognorm_mu_scan_needle_mins_ivt',
            'lognorm_sigma_scan_needle_mins_ivt', 
            'lognorm_mu_onset_arrival_mins_mt', 
            'lognorm_sigma_onset_arrival_mins_mt', 
            'lognorm_mu_arrival_scan_arrival_mins_mt', 
            'lognorm_sigma_arrival_scan_arrival_mins_mt',
            'lognorm_mu_scan_puncture_mins_mt',
            'lognorm_sigma_scan_puncture_mins_mt', 
            # Only used in sanity checks:
            'proportion2_of_mask1_with_onset_to_arrival_on_time_ivt',
            'proportion3_of_mask2_with_arrival_to_scan_on_time_ivt',
            'proportion4_of_mask3_with_onset_to_scan_on_time_ivt',
            'proportion5_of_mask4_with_enough_time_to_treat_ivt',
            'proportion1_of_all_with_onset_known_mt',
            'proportion4_of_mask3_with_onset_to_scan_on_time_mt',
            'proportion5_of_mask4_with_enough_time_to_treat_mt',
            ]
        expected_dtypes = [['float']] * len(keys)

        success = True
        for k, key in enumerate(keys):
            # Does this key exist?
            try:
                value_here = hospital_data[key]
                # Are those values of the expected data type?
                dtype_here = np.dtype(type(value_here))#.dtype)
                expected_dtypes_here = [
                    np.dtype(d) for d in expected_dtypes[k]
                    ]
                if dtype_here not in expected_dtypes_here:
                    print(''.join([
                        f'{key} is type {dtype_here} instead of ',
                        f'expected type {expected_dtypes_here}.'
                        ]))
                    success = False
                else:
                    pass  # The data is OK.
            except KeyError:
                print(f'{key} is missing from the hospital data.')
                success = False
        if success is False:
            error_str = 'The input hospital data needs fixing.'
            raise ValueError(error_str) from None
        else:
            pass  # All of the data is ok.


    def _fudge_patients_after_time_limit(
            self,
            patient_times_mins: float,
            proportion_within_limit: float,
            time_limit_mins: float
            ):
        """
        Make sure the proportion of patients with arrival time
        below X hours matches the hospital performance data.
        Set a few patients now to have arrival times above
        X hours.
        
        Inputs:
        -------
        patient_times_mins      - array. One time per patient.
        proportion_within_limit - float. How many of those times should
                                  be under the time limit?
        time_limit_mins         - float. The time limit we're comparing
                                  these patient times with.
                                
        Returns:
        --------
        patient_times_mins - array. The input time array but with some
                             times changed to be past the limit so that
                             the proportion within the limit is met.
        """
        # Only do this step if the proportion is within the allowed
        # bounds.
        if ((proportion_within_limit > 1.0) |
            (proportion_within_limit < 0.0) |
            (np.isnan(proportion_within_limit))):
            return patient_times_mins
        # How many patients are currently arriving after the limit?
        inds_times_after_limit = np.where(
            patient_times_mins > time_limit_mins)[0]
        n_times_after_limit = len(inds_times_after_limit)
        
        # How many patients should arrive after the limit?
        expected_times_after_limit = np.sum(np.random.binomial(
                1,                                # Number of trials
                (1.0 - proportion_within_limit),  # Probability of success
                len(patient_times_mins)           # Number of samples drawn
                ))
        
        # How many extra patients should we alter the values for?
        n_times_to_fudge = expected_times_after_limit - n_times_after_limit
        
        if n_times_to_fudge > 0:
            if len(np.where(patient_times_mins <= time_limit_mins)[0]) > 0:
                # Randomly pick this many patients out of those who are
                # currently arriving within the limit.
                inds_times_to_fudge = np.random.choice(
                    np.where(patient_times_mins <= time_limit_mins)[0],
                    size=n_times_to_fudge,
                    replace=False
                    )
                # Set these patients to be beyond the thrombectomy time limit.
                patient_times_mins[inds_times_to_fudge] = self.limit_mt_mins + 1
        return patient_times_mins


    def _sanity_check_masked_patient_proportions(
            self, leeway: float=0.25):
        """
        Check if generated proportions match the targets.
        
        Compare proportions of patients passing each mask with
        the target proportions (e.g. from real hospital performance 
        data).
        
        Inputs:
        -------
        leeway - float. How far away the generated proportion can be
                 from the target proportion without raising a warning.
                 Set this to 0.0 for no difference allowed (not 
                 recommended!) or >1.0 to guarantee no warnings.
        """
        # Only bother with these checks if there are enough patients
        # in the array.
        if self.patients_per_run <= 30:
            return
        elif np.all(self.trial['stroke_type_code'].data == 0):
            # All patients are "other" stroke type so we don't care
            # so much about the occlusion pathway.
            return
        
        target_proportions = [
            'proportion1_of_all_with_onset_known_',
            'proportion2_of_mask1_with_onset_to_arrival_on_time_',
            'proportion3_of_mask2_with_arrival_to_scan_on_time_',
            'proportion4_of_mask3_with_onset_to_scan_on_time_',
            'proportion5_of_mask4_with_enough_time_to_treat_',
            'proportion6_of_mask5_with_treated_'
        ]
        for i, treatment in enumerate(['ivt', 'mt']):
            for j, proportion_name in enumerate(target_proportions):
                try:
                    target_proportion = self.target_data_dict[
                        proportion_name + treatment]
                    success = True
                except KeyError:
                    # This proportion hasn't been given by the user.
                    success = False
                    
                if success is False:
                    pass  # Don't perform any checks.
                else:
                    generated_proportion = self.trial_performance_dict[
                        proportion_name + treatment]
                    
                    # If there's a problem, print this label:
                    label_to_print = proportion_name.replace('_', ' ')
                    label_to_print = label_to_print.split('proportion')[1] 
                    label_to_print += treatment

                    # Compare with target proportion:
                    self._check_proportion(
                        generated_proportion,
                        target_proportion,
                        label=label_to_print,
                        leeway=leeway
                        )


    def _check_proportion(
            self,
            prop_current: float,
            prop_target: float,
            label: str='',
            leeway: float=0.1
            ):
        """
        Check whether generated proportion is close to the target.
        
        If the proportion is more than (leeway*100)% off the target,
        raise a warning message.
        
        Inputs:
        prop_current - float. Calculated proportion between 0.0 and 1.0.
        prop_target  - float. Target proportion between 0.0 and 1.0.
        label        - str. Label to print if there's a problem.
        leeway       - float. How far away the calculated proportion is
                       allowed to be from the target.
        """
        if ((prop_current > prop_target + leeway) or
            (prop_current < prop_target - leeway)):
            print(''.join([
                f'The proportion of "{label}" is ',
                f'over {leeway*100}% out from the target value. '
                f'Target: {prop_target:.5f}, ',
                f'current: {prop_current:.5f}.'
                ]))
        else:
            pass


    def _sanity_check_distribution_statistics(
            self,
            patient_times: np.ndarray,
            mu_target: float,
            sigma_target: float,
            mu_generated: float,
            sigma_generated: float,
            label: str=''
            ):
        """
        Check whether generated times follow target distribution.
        
        Raise warning if:
        - the new mu is outside the old mu +/- old sigma, or
        - the new sigma is considerably larger than old sigma.
        
        Inputs:
        -------
        patient_times   - np.ndarray. The distribution to check.
        mu_target       - float. mu for the target distribution.
        sigma_target    - float. sigma for the target distribution.
        mu_generated    - float. mu to check.
        sigma_generated - float. sigma to check.
        label           - str. Label to print if there's a problem.
        """
        # Only bother with these checks if there are enough patients
        # in the array and the sigma is not zero.
        if len(patient_times) <= 30:
            return
        elif sigma_target < 1e-5:
            return
        
        # Check generated mu:
        if abs(mu_target - mu_generated) > sigma_target:
            print(''.join([
                f'Warning: the log-normal "{label}" distribution ',
                'has a mean outside the target mean plus or minus ',
                'one standard deviation.'
            ]))
        else:
            pass

        # Check generated sigma:
        if sigma_target > 3*sigma_generated:
            print(''.join([
                f'Warning: the log-normal "{label}" distribution ',
                'has a standard deviation at least 3 times as large ',
                'as the target standard deviation.'
            ]))
        else:
            pass


    # ##################################
    # ######## STROKE TYPE CODE ########
    # ##################################

    def _assign_stroke_type_code(self):
        """
        Assign stroke type based partly on treatment decision.

        Available combinations:
        +------+------+--------------+--------------|
        | Type | Code | Thrombolysis | Thrombectomy |
        +------+------+--------------+--------------|
        | nLVO | 1    | Yes or no    | No           |
        | LVO  | 2    | Yes or no    | Yes or no    |
        | Else | 0    | No           | No           |
        +------+------+--------------+--------------|
        
        --- Requirements ---
        The patient cohort can be split into the following four groups:

        ▓A▓ - patients receiving thrombectomy only.
        ░B░ - patients receiving thrombolysis and thrombectomy.
        ▒C▒ - patients receiving thrombolysis only.
        █D█ - patients receiving neither thrombolysis nor thrombectomy.

        For example, the groups might have these proportions:
         A   B       C                             D
        ▓▓▓░░░░░▒▒▒▒▒▒▒▒▒▒▒████████████████████████████████████████████

        The rules:
        + Groups ▓A▓ and ░B░ must contain only LVOs.
        + Group ▒C▒ must contain only nLVOs and LVOs.
        + Group █D█ may contain any stroke type.

        --- Method ---
        1. Set everyone in Groups ▓A▓ and ░B░ to have LVOs.
        2. Decide Groups ░B░ and ▒C▒ combined should contain LVO and
           nLVO patients in the same proportion as the whole cohort.
           For example:   B       C          Some LVO patients
                        ░░░░░▒▒▒▒▒▒▒▒▒▒▒     have already been placed
                        <-LVO--><-nLVO->     into Group ░B░.

           Calculate how many patients should have nLVO according to
           the target proportion. Set the number of nLVO patients in
           Group ▒C▒ to either this number or the total number of 
           patients in Group ▒C▒, whichever is smaller. Then
           the rest of the patients in Group ▒C▒ are assigned as LVO.
           The specific patients chosen for each are picked at random.
        3. Randomly pick patients in Group █D█ to be each stroke type
           so that the numbers add up as expected.

        --- Result ---
        Creation of self.trial['stroke_type_code'] array.
        """
        # Initially set all patients to "other stroke type":
        trial_stroke_type_code = np.full(self.patients_per_run, 0, dtype=int)
        # Keep track of which patients we've assigned a value to:
        trial_type_assigned_bool = np.zeros(self.patients_per_run, dtype=int)
        
        # Target numbers of patients with each stroke type:
        total = dict()
        total['total'] = self.patients_per_run
        total['lvo'] = np.sum(np.random.binomial(
            1,                    # Number of trials
            self.proportion_lvo,  # Probability of success
            total['total']        # Number of samples drawn
            ))
        # If this generated number of nLVO patients is too high,
        # instead set it to the number of patients that do not
        # have LVOs. Then the number of "other" patients will be zero.
        total['nlvo'] = np.minimum(
            np.sum(np.random.binomial(
                1,                     # Number of trials
                self.proportion_nlvo,  # Probability of success
                total['total']         # Number of samples drawn
                )),
            total['total'] - total['lvo']
            )
        total['other'] = total['total'] - (total['lvo'] + total['nlvo'])

        # Find which patients are in each group.
        inds_groupA = np.where(
            (self.trial['mt_chosen_bool'].data == 1) & 
            (self.trial['ivt_chosen_bool'].data == 0))[0]
        inds_groupB = np.where(
            (self.trial['ivt_chosen_bool'].data == 1) &
            (self.trial['mt_chosen_bool'].data == 1))[0]
        inds_groupBC = np.where(
            self.trial['ivt_chosen_bool'].data == 1)[0]   # Groups B & C
        inds_groupC = np.where(
            (self.trial['ivt_chosen_bool'].data == 1) &
            (self.trial['mt_chosen_bool'].data == 0))[0]
        
        # Find how many patients in each group have each stroke type.
        # Step 1: all thrombectomy patients are LVO.
        groupA = dict(
            total = len(inds_groupA),
            lvo = len(inds_groupA),
            nlvo = 0,
            other = 0,
            )
        groupB = dict(
            total = len(inds_groupB),
            lvo = len(inds_groupB),
            nlvo = 0,
            other = 0
            )
        
        # If this is more LVO than we expect, juggle the initial 
        # numbers.  
        if groupB['lvo'] + groupA['lvo'] > total['lvo']:
            # Set the total expected LVO to the new required 
            # amount and take the same amount out of the nLVO group.
            diff = groupB['lvo'] + groupA['lvo'] - total['lvo']
            total['lvo'] += diff
            total['nlvo'] -= diff
            if total['nlvo'] < 0:
                # If that pushes nLVO below zero, instead take the 
                # difference from the "other" group.
                total['other'] -= total['nlvo']
                total['nlvo'] = 0
            else: pass
        else: pass
        
        # Step 2: all thrombolysis patients have nLVO or LVO
        # in the same ratio as the patient proportions if possible.
        # Work out how many people have nLVO and were thrombolysed.
        # This is the smaller of: 
        # the total number of people in group C...
        n1 = len(inds_groupC)
        # ... and the number of people in groups B and C that 
        # should have nLVO according to the target proportion...
        n2 = int(
            round(len(inds_groupBC) * (total['nlvo']/total['total']), 0)
            )
        # ... to give this value:
        n_nlvo_and_thrombolysis = np.minimum(n1, n2)
        
        groupC = dict(
            total = len(inds_groupC),
            lvo = len(inds_groupC) - n_nlvo_and_thrombolysis,
            nlvo = n_nlvo_and_thrombolysis,
            other = 0
            )
        
        # Sanity check - are there enough LVO patients left?
        if groupA['lvo'] + groupB['lvo'] + groupC['lvo'] > total['lvo']:
            lvo_before = groupC['lvo']
            # Set the number of LVO patients here to be exactly the
            # number not assigned to either groups A or B.
            groupC['lvo'] = total['lvo'] - (groupA['lvo'] + groupB['lvo'])
            # Add the difference back onto the nLVO group.
            groupC['nlvo'] = groupC['nlvo'] + (lvo_before - groupC['lvo'])
                
        # Randomly select which patients in Group C have each type.
        # For LVO, select these people... 
        inds_lvo_groupC = np.random.choice(
            inds_groupC,
            size=groupC['lvo'],
            replace=False
            )
        # ... and for nLVO, select everyone else.
        inds_nlvo_groupC = np.array(list(
            set(inds_groupC) -
            set(inds_lvo_groupC)
            ), dtype=int)
        
        # Bookkeeping:
        # Set the chosen patients to their stroke types:
        trial_stroke_type_code[inds_groupA] = 2
        trial_stroke_type_code[inds_groupB] = 2
        trial_stroke_type_code[inds_lvo_groupC] = 2
        trial_stroke_type_code[inds_nlvo_groupC] = 1
        
        # Keep track of which patients have been assigned a type:
        trial_type_assigned_bool[inds_groupA] += 1
        trial_type_assigned_bool[inds_groupB] += 1
        trial_type_assigned_bool[inds_groupC] += 1

        # Step 3: everyone else is in Group D.
        groupD = dict(
            total = (total['total'] - 
                    (groupA['total'] + groupB['total'] + groupC['total'])),
            lvo = (total['lvo'] - 
                  (groupA['lvo'] + groupB['lvo'] + groupC['lvo'])),
            nlvo = (total['nlvo'] - 
                   (groupA['nlvo'] + groupB['nlvo'] + groupC['nlvo'])),
            other = (total['other'] - 
                    (groupA['other'] + groupB['other'] + groupC['other']))
            )
        
        
        # For each stroke type, randomly select some indices out
        # of those that have not yet been assigned a stroke type.
        # LVO selects from everything in Group D:
        inds_groupD_lvo = np.random.choice(
            np.where(trial_type_assigned_bool == 0)[0],
            size=groupD['lvo'],
            replace=False
            )
        trial_type_assigned_bool[inds_groupD_lvo] += 1
        trial_stroke_type_code[inds_groupD_lvo] = 2
        
        # nLVO selects from everything in Group D that hasn't
        # already been assigned to LVO:
        inds_groupD_nlvo = np.random.choice(
            np.where(trial_type_assigned_bool == 0)[0],
            size=groupD['nlvo'],
            replace=False
            )
        trial_type_assigned_bool[inds_groupD_nlvo] += 1
        trial_stroke_type_code[inds_groupD_nlvo] = 1

        # Other types select from everything in Group D that hasn't
        # already been assigned to LVO or nLVO:
        inds_groupD_other = np.random.choice(
            np.where(trial_type_assigned_bool == 0)[0],
            size=groupD['other'],
            replace=False
            )
        trial_type_assigned_bool[inds_groupD_other] += 1

        # ### Final check ###
        # Sanity check that each patient was assigned exactly 
        # one stroke type:
        if np.any(trial_type_assigned_bool > 1):
            print(''.join([
                'Warning: check the stroke type code assignment. ',
                'Some patients were assigned multiple stroke types. ']))
        if np.any(trial_type_assigned_bool == 0):
            print(''.join([
                'Warning: check the stroke type code assignment. ',
                'Some patients were not assigned a stroke type. ']))

        # Now store this final array in self:
        self.trial['stroke_type_code'].data = trial_stroke_type_code
