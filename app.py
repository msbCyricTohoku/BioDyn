#BioDyn
#Developed by Mehrdad S. Beni -- May 2025
#import config #import the configuration file
import yaml
import sys
import os
import io
import datetime
from flask import Flask, request, render_template, redirect, url_for, flash, send_file
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib
matplotlib.use('Agg') # <--set backend BEFORE importing pyplot
import matplotlib.pyplot as plt
import traceback
from scipy.stats import pearsonr


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#here we load the yaml file
def load_config_from_yaml(filepath='config.yaml'):
    print(f"--- Loading settings from {filepath} ---")

    required_keys = [
        'DATASET_PATH', 'INTERMEDIATE_RESULTS_PATH', 'AGE_COLUMN_NAME',
        'BIOMARKER_FEATURES','USE_PCA','DEFAULT_BIOMARKER_VALUES',
        'DEFAULT_AGE_FORM_VALUE','MIN_S_VALUE','MIN_DENOMINATOR', 
        'MIN_VARIANCE','MIN_R_CHAR','S2_BA_FLOOR','SIG_CORR_VAL',
        'SIG_P_THRESH','PLOTS_OUTPUT_DIR','SAVE_TRAINING_BA_VALUES',
        'SAVE_PLOTS_ON_TRAIN','PLOT_FILENAME_TIMESTAMP', 'plot_labels'
    ]

    required_plot_label_keys = [
        'plot1_title', 'plot1_xlabel', 'plot1_ylabel', 'plot2_title',
        'plot2_xlabel', 'plot2_ylabel', 'plot3_title', 'plot3_xlabel', 'plot3_ylabel'
    ]

    try:
        with open(filepath, 'r') as f:
            config_data = yaml.safe_load(f)

        if not isinstance(config_data, dict):
             print(f"FATAL ERROR: Content of '{filepath}' is not a valid YAML dictionary.")
             sys.exit(1) #exit if structure is wrong

        missing_keys = [key for key in required_keys if key not in config_data]
        
        if missing_keys:
            print(f"FATAL ERROR: Missing required keys in '{filepath}': {', '.join(missing_keys)}")
            sys.exit(1)

        plot_labels = config_data['plot_labels']
        
        if not isinstance(plot_labels, dict):
            print(f"FATAL ERROR: 'plot_labels' in '{filepath}' must be a dictionary.")
            sys.exit(1)
        missing_plot_keys = [key for key in required_plot_label_keys if key not in plot_labels]
        
        if missing_plot_keys:
            print(f"FATAL ERROR: Missing required keys in 'plot_labels' in '{filepath}': {', '.join(missing_plot_keys)}")
            sys.exit(1)

        if not isinstance(config_data['BIOMARKER_FEATURES'], list) or not config_data['BIOMARKER_FEATURES']:
            print(f"FATAL ERROR: 'BIOMARKER_FEATURES' in '{filepath}' must be a non-empty list.")
            sys.exit(1)
        if not isinstance(config_data['DEFAULT_BIOMARKER_VALUES'], dict):
             print(f"FATAL ERROR: 'DEFAULT_BIOMARKER_VALUES' in '{filepath}' must be a dictionary.")
             sys.exit(1)
        
        missing_defaults = [f for f in config_data['BIOMARKER_FEATURES'] if f not in config_data['DEFAULT_BIOMARKER_VALUES']]
        
        if missing_defaults:
             print(f"Configuration Warning in '{filepath}': Missing default values for biomarkers: {', '.join(missing_defaults)}. App will continue but defaults may be incomplete.")
             
        print(f"Settings loaded successfully from {filepath}.")
        
        return config_data

    except FileNotFoundError:
        
        print(f"FATAL ERROR: Configuration file '{filepath}' not found. Cannot start application.")
        
        sys.exit(1) #exit if config file is essential and missing
    
    except yaml.YAMLError as e:
        
        print(f"FATAL ERROR: Could not parse configuration file '{filepath}': {e}")
        
        sys.exit(1) #exit on parsing error
    
    except Exception as e:
        
        print(f"FATAL ERROR: An unexpected error occurred loading configuration from '{filepath}': {e}")
        
        sys.exit(1) #exit on other errors

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#timestamp function
def get_timestamp_string():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#load config
APP_CONFIG = load_config_from_yaml()

#global variable definitions
DATASET_PATH = APP_CONFIG['DATASET_PATH']
INTERMEDIATE_RESULTS_PATH = APP_CONFIG['INTERMEDIATE_RESULTS_PATH']
AGE_COLUMN_NAME = APP_CONFIG['AGE_COLUMN_NAME']
FEATURE_COLUMNS = APP_CONFIG['BIOMARKER_FEATURES']
USE_PCA = APP_CONFIG['USE_PCA']
DEFAULT_BIOMARKER_VALUES = APP_CONFIG['DEFAULT_BIOMARKER_VALUES']
DEFAULT_AGE_FORM_VALUE = APP_CONFIG['DEFAULT_AGE_FORM_VALUE']


#threshold constants
MIN_S_VALUE = float(APP_CONFIG['MIN_S_VALUE'])      #skip components with near-zero RMSE
MIN_DENOMINATOR = float(APP_CONFIG['MIN_DENOMINATOR'])     #avoid divide-by-tiny denominator
MIN_VARIANCE = float(APP_CONFIG['MIN_VARIANCE'])     #skip regressions on near-constant data
MIN_R_CHAR = float(APP_CONFIG['MIN_R_CHAR'])      #avoid invalid r_char sqrt(1-r^2)
S2_BA_FLOOR  = float(APP_CONFIG['S2_BA_FLOOR'])      #floor for S2_BA
SIG_CORR_VAL = float(APP_CONFIG['SIG_CORR_VAL'])      #set Pearson correlation coeff. for report
SIG_P_THRESH = float(APP_CONFIG['SIG_P_THRESH'])      #set P value threshold for report



PLOTS_OUTPUT_DIR = APP_CONFIG['PLOTS_OUTPUT_DIR']
SAVE_TRAINING_BA_VALUES = APP_CONFIG['SAVE_TRAINING_BA_VALUES']
SAVE_PLOTS_ON_TRAIN = APP_CONFIG['SAVE_PLOTS_ON_TRAIN']
PLOT_FILENAME_TIMESTAMP = APP_CONFIG['PLOT_FILENAME_TIMESTAMP']


#plot labels from yaml file
_plot_labels_dict = APP_CONFIG.get('plot_labels', {})
PLOT1_TITLE = _plot_labels_dict.get('plot1_title', 'Default Plot 1 Title')
PLOT1_XLABEL = _plot_labels_dict.get('plot1_xlabel', 'Default Plot 1 X-Label')
PLOT1_YLABEL = _plot_labels_dict.get('plot1_ylabel', 'Default Plot 1 Y-Label')
PLOT2_TITLE = _plot_labels_dict.get('plot2_title', 'Default Plot 2 Title')
PLOT2_XLABEL = _plot_labels_dict.get('plot2_xlabel', 'Default Plot 2 X-Label')
PLOT2_YLABEL = _plot_labels_dict.get('plot2_ylabel', 'Default Plot 2 Y-Label')
PLOT3_TITLE = _plot_labels_dict.get('plot3_title', 'Default Plot 3 Title')
PLOT3_XLABEL = _plot_labels_dict.get('plot3_xlabel', 'Default Plot 3 X-Label')
PLOT3_YLABEL = _plot_labels_dict.get('plot3_ylabel', 'Default Plot 3 Y-Label')

#load_settings_from_yaml()

#KDM BA_EC calculations start here
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def calculate_ba_e(x_values, q_values, k_values, s_values):

    if not isinstance(x_values, (list, np.ndarray)) or not isinstance(q_values, list) or not isinstance(k_values, list) or not isinstance(s_values, list): 
        return np.nan
    if not all(len(lst) == len(x_values) for lst in [q_values, k_values, s_values]):
        return np.nan

    numerator = 0; denominator = 0; num_components = len(x_values)

    for j in range(num_components):
        if j >= len(k_values) or j >= len(s_values) or j >= len(q_values): 
            continue
        if s_values[j] is None or np.isnan(s_values[j]) or s_values[j] < MIN_S_VALUE: 
            continue
        if k_values[j] is None or np.isnan(k_values[j]): 
            continue
        if q_values[j] is None or np.isnan(q_values[j]): 
            continue
        try:
            weight = k_values[j] / (s_values[j] ** 2)

            term = (x_values[j] - q_values[j]) * weight
            
            denominator_term = (k_values[j] / s_values[j]) ** 2
            
            if np.isnan(term) or np.isnan(denominator_term): 
                continue
            
            numerator += term; 
            denominator += denominator_term

        except (ValueError, TypeError, ZeroDivisionError): 
            continue
            
    if abs(denominator) < MIN_DENOMINATOR: 
        return np.nan
    return numerator / denominator

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def calculate_ba_ec(x_values, q_values, k_values, s_values, ca, s2_ba):
    if not isinstance(x_values, (list, np.ndarray)) or not isinstance(q_values, list) or not isinstance(k_values, list) or not isinstance(s_values, list) or not isinstance(ca, (int, float)): 
        return np.nan
    
    numerator = 0; denominator = 0; num_components = len(x_values)
    
    for j in range(num_components):
        if j >= len(k_values) or j >= len(s_values) or j >= len(q_values): 
            continue
        if s_values[j] is None or np.isnan(s_values[j]) or s_values[j] < MIN_S_VALUE: 
            continue
        if k_values[j] is None or np.isnan(k_values[j]): 
            continue
        if q_values[j] is None or np.isnan(q_values[j]): 
            continue
        try:
            weight = k_values[j] / (s_values[j] ** 2)
            
            term = (x_values[j] - q_values[j]) * weight
            
            denominator_term = (k_values[j] / s_values[j]) ** 2
            
            if np.isnan(term) or np.isnan(denominator_term): 
                continue
            
            numerator += term; 
            denominator += denominator_term
        
        except (ValueError, TypeError, ZeroDivisionError): 
            continue
    if s2_ba is not None and not np.isnan(s2_ba) and s2_ba > 1e-9:
        try:
            if np.isnan(ca): return np.nan
            numerator += ca / s2_ba; 
            denominator += 1 / s2_ba

            #print("GGGGGGG -->",ca / s2_ba)
            #print("GGGGGGG -->",1 / s2_ba)

        except (ValueError, TypeError, ZeroDivisionError): return np.nan

    if abs(denominator) < MIN_DENOMINATOR: 
        return np.nan

    return numerator / denominator #1e-9 tolerance was used to avoid divide by small number or zero that leads to blowup (change if needed)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def calculate_r_char(r_values):
    
    numerator = 0
    denominator = 0
    valid_r_count = 0

    for r in r_values:
        if r is None or np.isnan(r) or abs(r) >= 1: 
            continue
        
        sqrt_term_val = 1 - r**2

        if sqrt_term_val <= MIN_R_CHAR: 
            continue

        sqrt_term = np.sqrt(sqrt_term_val)

        numerator += (r**2) / sqrt_term
        denominator += r / sqrt_term
        valid_r_count += 1

    if abs(denominator) < 1e-9 or valid_r_count == 0: 
        return 0.0

    r_char_val = numerator / denominator

    #print("r-loop:", valid_r_count)

    return np.clip(r_char_val, -0.9999, 0.9999) #clipped to esure it is bounded between -1 to 1 (if needed change)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def calculate_s2_ba(ba_e_list, ca_list, r_char, m):
    #this is the corrected s2_ba function, original version has mistake and did not match with Levine equations
    valid_indices = [i for i, (bae, ca) in enumerate(zip(ba_e_list, ca_list))
                     if bae is not None and not np.isnan(bae) and
                        ca is not None and not np.isnan(ca)]

    if not valid_indices or len(valid_indices) < 2:
        print("[S2_BA Calc - Pop] Insufficient valid data.")
        return np.nan

    ba_e_list_valid = np.array([ba_e_list[i] for i in valid_indices])
    ca_list_valid = np.array([ca_list[i] for i in valid_indices])
    n = len(ba_e_list_valid)

    #parameter checks
    if n < 2 or m <= 0 or r_char is None or np.isnan(r_char) or abs(r_char) < MIN_R_CHAR:
        print(f"[S2_BA Calc - Pop] Invalid input params: n={n}, m={m}, r_char={r_char}. Cannot divide by r_char^2.")
        return np.nan
    if abs(r_char) >= 1:
         print(f"[S2_BA Calc - Pop] Invalid r_char value: {r_char}.")
         return np.nan

    #term 1 calculation (here we use population variance like Levine paper but sample is better -- observed not much diffs.)
    ba_minus_ca = ba_e_list_valid - ca_list_valid
    #here we use ddof=0 for population variance (divides by n) -- tried manually too same thing -- change to 1 for sample
    var_ba_minus_ca_pop = np.var(ba_minus_ca, ddof=0)

    term1 = var_ba_minus_ca_pop

    #term 2 calculation
    ca_range = np.max(ca_list_valid) - np.min(ca_list_valid)
    
    #expanded from the inline conditional:
    if ca_range < 1e-9:
        var_ca_approx = 0.0
    else:
        var_ca_approx = (ca_range ** 2) / 12.0


    try:
        factor_A = (1.0 - r_char**2) / (r_char**2)
        factor_B = var_ca_approx / m
        term2 = factor_A * factor_B
    except Exception as e:
        print(f"[S2_BA Calc - Pop] Error calculating Term 2: {e}")
        traceback.print_exc()
        return np.nan

    #final calculation (still flooring to avoid negative numver or divide by zero for certain data when model fails)
    result = term1 - term2
    
    if pd.notna(result) and result > 0:
        final_s2_ba = max(S2_BA_FLOOR, result)
    else:
        final_s2_ba = S2_BA_FLOOR

    return final_s2_ba

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



#flask app starts here
app = Flask(__name__)
the_key = '123456' #change this if deploy to a server outside LAN
app.secret_key = os.environ.get('FLASK_SECRET_KEY', the_key)

'''
DATASET_PATH = config.DATASET_PATH
INTERMEDIATE_RESULTS_PATH = config.INTERMEDIATE_RESULTS_PATH
FEATURE_COLUMNS = config.BIOMARKER_FEATURES
USE_PCA = config.USE_PCA
AGE_COLUMN_NAME = config.AGE_COLUMN_NAME
PLOTS_OUTPUT_DIR = config.PLOTS_OUTPUT_DIR
PLOT1_TITLE = config.PLOT1_TITLE
PLOT1_XLABEL = config.PLOT1_XLABEL
PLOT1_YLABEL = config.PLOT1_YLABEL
PLOT2_TITLE = config.PLOT2_TITLE
PLOT2_XLABEL = config.PLOT2_XLABEL
PLOT2_YLABEL = config.PLOT2_YLABEL
PLOT3_TITLE = config.PLOT3_TITLE
PLOT3_XLABEL = config.PLOT3_XLABEL
PLOT3_YLABEL = config.PLOT3_YLABEL
'''


scaler = None; 
pca = None; 
q_values = []
k_values = [] 
s_values = [] 
r_values = [] 
r_char = 0; 
S2_BA = 0
last_training_stats = {}
last_training_failed = False
selected_indices = [] # <-- ADD THIS: Store indices of selected features/PCs
selected_labels = [] # <-- ADD THIS: Store labels of selected features/PCs (optional but good for logging)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#loading the dataset referred from config file
def load_dataset():
    try: 
        df = pd.read_csv(DATASET_PATH)
        df.columns = df.columns.str.strip()

    except FileNotFoundError: 
        print(f"Dataset file '{DATASET_PATH}' not found.")
        return pd.DataFrame(columns=FEATURE_COLUMNS + [AGE_COLUMN_NAME, 'source'])

    except Exception as e: 
        print(f"Error loading dataset: {e}.")
        return pd.DataFrame(columns=FEATURE_COLUMNS + [AGE_COLUMN_NAME, 'source'])

    if 'source' not in df.columns: df['source'] = 'existing'

    required_cols = FEATURE_COLUMNS + [AGE_COLUMN_NAME]
    
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:  
        for col in missing_cols: df[col] = np.nan
    for col in required_cols:
         if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    return df
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#the intermediate results in xlsx format
def load_intermediate_results():
    
    global q_values, k_values, s_values, r_values, r_char, S2_BA, scaler, pca, USE_PCA

    if not os.path.exists(INTERMEDIATE_RESULTS_PATH): 
        print(f"Intermediate results file not found.")
        return None
    try:
        xls = pd.ExcelFile(INTERMEDIATE_RESULTS_PATH)
        required_sheets = ['PCA_Info', 'Regression_Coefficients', 'Model_Parameters']

        if not all(sheet in xls.sheet_names for sheet in required_sheets): 
            print("Intermediate results missing sheets.")
            return None

        df_pca_info = pd.read_excel(xls, sheet_name='PCA_Info')
        
        df_coefficients = pd.read_excel(xls, sheet_name='Regression_Coefficients')
        
        df_parameters = pd.read_excel(xls, sheet_name='Model_Parameters')

        if 'USE_PCA' not in df_parameters.columns: 
            print("USE_PCA setting missing.")
            return None

        saved_use_pca = df_parameters['USE_PCA'].iloc[0]

        if 'feature' not in df_pca_info.columns: 
            print("feature column missing.")
            return None

        saved_features = df_pca_info['feature'].tolist()

        if saved_use_pca != USE_PCA or set(saved_features) != set(FEATURE_COLUMNS): 
            print("Config mismatch. Retraining required.")
            return None


        coeff_cols = ['q_values', 'k_values', 's_values', 'r_values']
        
        param_cols = ['r_char', 'S2_BA']

        if not all(col in df_coefficients.columns for col in coeff_cols): 
            print("Coeff sheet columns missing.")
            return None
        
        if not all(col in df_parameters.columns for col in param_cols): 
            print("Param sheet columns missing.")
            return None
        
        num_expected_coeffs = len(FEATURE_COLUMNS)
        
        if len(df_coefficients) != num_expected_coeffs: 
            print("Coeff count mismatch.")
            return None
        
        q_values[:] = df_coefficients['q_values'].tolist() 
        k_values[:] = df_coefficients['k_values'].tolist() 
        s_values[:] = df_coefficients['s_values'].tolist()
        r_values[:] = df_coefficients['r_values'].tolist()
        
        r_char = df_parameters['r_char'].iloc[0]

        S2_BA = df_parameters['S2_BA'].iloc[0]
        
        scaler = None
        pca = None
        
        print(f"Loaded intermediate results successfully (USE_PCA={saved_use_pca}).")
        
        return saved_features

    except Exception as e:
        print(f"Error loading intermediate results: {e}.")
        q_values[:], k_values[:], s_values[:], r_values[:] = [], [], [], []
        r_char, S2_BA = 0, 0; scaler, pca = None, None
        return None

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#saving intermediate results in xlsx format
def save_intermediate_results(df_training_results=None):

    global q_values, k_values, s_values, r_values, r_char, S2_BA, pca, scaler, USE_PCA, FEATURE_COLUMNS, last_training_stats
    
    if scaler is None or (USE_PCA and pca is None) or not all([isinstance(q_values, list), isinstance(k_values, list), isinstance(s_values, list), isinstance(r_values, list), len(k_values) == len(FEATURE_COLUMNS) ]): 
        print("Error: Cannot save intermediate results...")
        return

    #some prints for debug stuff
    #print(f"[SAVE XLS DEBUG] config.SAVE_TRAINING_BA_VALUES = {SAVE_TRAINING_BA_VALUES}")
    #if df_training_results is not None: 
        #print(f"[SAVE XLS DBG] df_training_results shape: {df_training_results.shape}")
        #print(f"[SAVE XLS DBG] df_training_results head:\n{df_training_results.head()}")
        #print(f"[SAVE XLS DBG] df_training_results NaNs:\n{df_training_results.isnull().sum()}")
    #else: 
        #print("[SAVE XLS DBG] df_training_results is None.")
    #print(f"[SAVE XLS DBG] Stats being saved: {last_training_stats}")
    
    try:
        with pd.ExcelWriter(INTERMEDIATE_RESULTS_PATH) as writer:
            pca_info = pd.DataFrame({'feature': FEATURE_COLUMNS, 'scaler_mean': scaler.mean_, 'scaler_scale': scaler.scale_})

            if USE_PCA and pca: pca_info['pca_components'] = [str(comp) for comp in pca.components_]

            pca_info['explained_variance_ratio'] = pca.explained_variance_ratio_ if hasattr(pca, 'explained_variance_ratio_') else [np.nan] * len(FEATURE_COLUMNS)
            
            pca_info.to_excel(writer, sheet_name='PCA_Info', index=False)
            
            coeff_label = 'PC' if USE_PCA else 'Feature'; 
            
            pd.DataFrame({ coeff_label: [f'{coeff_label}{i+1}' if USE_PCA else FEATURE_COLUMNS[i] for i in range(len(FEATURE_COLUMNS))], 'q_values': q_values, 'k_values': k_values, 's_values': s_values, 'r_values': r_values }).to_excel(writer, sheet_name='Regression_Coefficients', index=False)
            
            params_data = {'USE_PCA': [USE_PCA], 'r_char': [r_char], 'S2_BA': [S2_BA]}
            
            params_data.update(last_training_stats)
            
            pd.DataFrame(params_data).to_excel(writer, sheet_name='Model_Parameters', index=False)
            
            if SAVE_TRAINING_BA_VALUES and df_training_results is not None and not df_training_results.empty:

                 df_to_save = df_training_results.dropna(subset=['BA_E', 'BA_EC'], how='all').copy()

                 if not df_to_save.empty: 
                    df_to_save.to_excel(writer, sheet_name='Training_BA_Results', index=False)
                    #print(f"[SAVE XLS DEBUG] Saved {len(df_to_save)} rows to Training_BA_Results sheet.")

                 else: 
                    print("[SAVE XLS DBG] No valid BA rows found to save.")
            elif SAVE_TRAINING_BA_VALUES: 
                print("[SAVE XLS DBG] Skip saving BA results (flag True but no data).")
        print(f"Saved intermediate results successfully to {INTERMEDIATE_RESULTS_PATH} (USE_PCA={USE_PCA}).")
    except Exception as e: 
        print(f"Error saving intermediate results: {e}")

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#data merging when uploaded and with those existing ones
def merge_with_existing_data(df_uploaded, dataset_path):

    ts = get_timestamp_string() #get_timestamp_string is defined globally
    #print(f"[{ts}] [MERGE] Starting data merge process...")

    df_existing = load_dataset()
    print(f"[{ts}] [MERGE] Loaded existing data: {len(df_existing)} rows.")

    df_uploaded.columns = df_uploaded.columns.str.strip() # Clean column names
    df_uploaded['source'] = 'uploaded' # Mark source
    print(f"[{ts}] [MERGE] Prepared uploaded data: {len(df_uploaded)} rows. Columns: {df_uploaded.columns.tolist()}")

    #make sure FEATURE_COLUMNS and AGE_COLUMN_NAME are accessible here (they are global)
    all_expected_columns = FEATURE_COLUMNS + [AGE_COLUMN_NAME, 'source']

    #align dataframes
    if 'source' not in df_existing.columns:
        df_existing['source'] = 'existing'

    df_existing_aligned = df_existing.reindex(columns=all_expected_columns, fill_value=np.nan)
    df_uploaded_aligned = df_uploaded.reindex(columns=all_expected_columns, fill_value=np.nan)

    #convert relevant columns to numeric in uploaded data (handle errors)
    for col in FEATURE_COLUMNS + [AGE_COLUMN_NAME]:
        if col in df_uploaded_aligned.columns: #check if present
            
            original_dtype = df_uploaded_aligned[col].dtype
            
            df_uploaded_aligned[col] = pd.to_numeric(df_uploaded_aligned[col], errors='coerce')
            
            if df_uploaded_aligned[col].isnull().any() and not pd.api.types.is_numeric_dtype(original_dtype):
                 print(f"[{ts}] [MERGE] Coerced non-numeric values to NaN in uploaded data column '{col}'.")
        # else: Column is missing, already filled with NaN by reindex


    df_combined = pd.concat([df_existing_aligned, df_uploaded_aligned], ignore_index=True)
    #print(f"[{ts}] [MERGE] Concatenated dataframes. Total rows before cleaning: {len(df_combined)}")

    df_combined.dropna(subset=FEATURE_COLUMNS + [AGE_COLUMN_NAME], how='all', inplace=True)
    #print(f"[{ts}] [MERGE] Total rows after dropping rows with all NaN in features/age: {len(df_combined)}")

    save_successful = False #bool to check if fail or success
    try:
        df_combined.to_csv(dataset_path, index=False)
        #will be called upon successful save
        print(f"[{ts}] [MERGE] Merged data successfully saved to '{dataset_path}'. Final rows: {len(df_combined)}")
        flash("Data merged successfully.", 'success')
        save_successful = True #success

    except Exception as e:
        #if combining fails error will shows for easy DBG
        print(f"[{ts}] [MERGE] ERROR saving merged data to '{dataset_path}': {e}")
        traceback.print_exc()
        flash(f"Error saving merged data: {e}", 'danger')
        save_successful = False #mark failure (already False, but explicit)

    #return status
    return save_successful

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#here we process the plots to be saved in the dir specified in the config file
def generate_and_save_plots(df_results, output_dir):

    global last_training_stats, AGE_COLUMN_NAME
    
    if df_results is None or df_results.empty: 
        print("[PLOT GEN] No data received.")
        return
    
    #print(f"[PLOT GEN] Generating plots and saving to {output_dir}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    required_plot_cols = [AGE_COLUMN_NAME, 'BA_EC', 'BA_Accel'] #check required cols
    
    if not all(col in df_results.columns for col in required_plot_cols): 
        print(f"[PLOT GEN] Error: Missing required columns.")
        return
    #print(f"[PLOT GEN DBG] df_results shape received: {df_results.shape}") 
    #print(f"[PLOT GEN DBG] df_results NaNs received:\n{df_results.isnull().sum()}") 
    
    df_plot = df_results.dropna(subset=required_plot_cols).copy()
    
    #print(f"[PLOT GEN DBG] df_plot shape after dropna: {df_plot.shape}") 
    
    if df_plot.empty: 
        print("[PLOT GEN] No valid data rows remaining for plotting.")
        return
    
    ca = df_plot[AGE_COLUMN_NAME]
    ba_ec = df_plot['BA_EC']
    ba_accel = df_plot['BA_Accel']
    
    mae_str = f"MAE={last_training_stats.get('MAE_BAEC_vs_CA', 'N/A'):.2f}" if isinstance(last_training_stats.get('MAE_BAEC_vs_CA'), (int, float)) else ""
    
    corr_str = f"Corr={last_training_stats.get('Corr_BAEC_vs_CA', 'N/A'):.2f}" if isinstance(last_training_stats.get('Corr_BAEC_vs_CA'), (int, float)) else ""
    
    stats_title = f"({mae_str}, {corr_str})" if mae_str or corr_str else ""
    
    ts = f"_{get_timestamp_string()}" if PLOT_FILENAME_TIMESTAMP else ""

    
    
    try:
        #plot 1: CA vs BA_EC -- u can modify these to change the plot style and stuff if needed
        plt.figure(figsize=(8, 8)); 
        
        plt.scatter(ca, ba_ec, alpha=0.6, s=40, label=f'Data (n={len(df_plot)})')
    
        plot_min = np.nanmin([ca.min(), ba_ec.min()]); 
        
        plot_max = np.nanmax([ca.max(), ba_ec.max()])
    
        if pd.notna(plot_min) and pd.notna(plot_max) and plot_min < plot_max: 
            plt.plot([plot_min, plot_max], [plot_min, plot_max], color='grey', linestyle='--', label='CA = BA_EC')
    
        plt.title(f'{PLOT1_TITLE}\n{stats_title}'); 
        
        #plt.xlabel(f'Chronological Age ({AGE_COLUMN_NAME})'); 
        plt.xlabel(PLOT1_XLABEL)
        
        plt.ylabel(PLOT1_YLABEL)
    
        plt.legend(); plt.grid(True, linestyle='--', alpha=0.6); 
        
        plt.tight_layout(); 
        
        filename = os.path.join(output_dir, f'ca_vs_ba_ec_scatter{ts}.png'); 
        
        plt.savefig(filename); 
        
        #plt.close(); print(f"[PLOT GEN] Saved: {filename}")
    
        #plot 2: CA vs BA Acceleration
        plt.figure(figsize=(8, 6)); 
        
        plt.scatter(ca, ba_accel, alpha=0.6, s=40)
        
        plt.axhline(0, color='grey', linestyle='--')
    
        plt.title(PLOT2_TITLE)
        
        plt.xlabel(PLOT2_XLABEL)
        
        plt.ylabel(PLOT2_YLABEL)
    
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout(); filename = os.path.join(output_dir, f'ca_vs_ba_accel_scatter{ts}.png')
        
        plt.savefig(filename); plt.close()
        
        #print(f"[PLOT GEN] Saved: {filename}")
    
        #plot 3: BA Acceleration Distribution
        plt.figure(figsize=(8, 6)); 
        
        ba_accel.plot(kind='hist', bins=30, alpha=0.7, density=True, label='Histogram')
        
        ba_accel.plot(kind='kde', color='red', label='Density Curve')
    
        mean_accel = ba_accel.mean()
        
        plt.axvline(mean_accel, color='black', linestyle='--', label=f'Mean = {mean_accel:.2f}')
    
        plt.title(PLOT3_TITLE)
        
        plt.xlabel(PLOT3_XLABEL)
        
        plt.ylabel(PLOT3_YLABEL)
    
        plt.legend(); 
        
        plt.grid(True, linestyle='--', alpha=0.6); 
        
        plt.tight_layout(); filename = os.path.join(output_dir, f'ba_accel_distribution{ts}.png')
        
        plt.savefig(filename)
        
        plt.close()
        
        #print(f"[PLOT GEN] Saved: {filename}")
    except Exception as e: print(f"[PLOT GEN] Error during plot saving: {e}"); plt.close()

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def write_significance_report(predictor_results, num_analysis_rows, num_predictors_analyzed,
                              predictor_type_desc, output_filename="BioDyn-report.txt",
                              sig_corr_threshold=SIG_CORR_VAL, sig_p_threshold=SIG_P_THRESH):

    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] [REPORT] Generating significance report...")

    # 1) filter for significant predictors
    significant = []
    for res in predictor_results:
        r_val = res.get('r')
        p_val = res.get('p')
        if isinstance(r_val, (int, float)) and isinstance(p_val, (int, float)) and \
           pd.notna(r_val) and pd.notna(p_val):
            if abs(r_val) > sig_corr_threshold and p_val < sig_p_threshold:
                significant.append(res)

    try:
        with open(output_filename, "w") as f:
            # header
            f.write(f"BioDyn Significance Report ({ts})\n")
            f.write("=" * 60 + "\n")
            f.write(f"Analysis based on: {num_analysis_rows} complete data rows.\n")
            f.write(f"Predictors Analyzed: {num_predictors_analyzed} ({predictor_type_desc})\n")
            f.write(f"Significance Criteria: |r| > {sig_corr_threshold} AND p < {sig_p_threshold}\n")
            f.write("-" * 60 + "\n\n")

            # 2) significant list
            if significant:
                f.write(f"Significant Predictors Found ({len(significant)}):\n")
                significant.sort(key=lambda x: abs(x.get('r', 0)), reverse=True)
                for pred in significant:
                    lbl = pred.get('label', 'Unknown')
                    r_val = pred.get('r', 0)
                    p_val = pred.get('p', 1)
                    f.write(f"- {lbl:<30} (r = {r_val:+.3f}, p = {p_val:.4g})\n")
                f.write("\nRecommendations:\n")
                f.write("  • Focus on these predictors for targeted analysis.\n")
                f.write("  • Update BIOMARKER_FEATURES in config.yaml or interpret these PCs if using PCA.\n")
            else:
                f.write("No predictors met the specified significance criteria.\n")

            # 3) overall correlation summary
            f.write("\nOverall Predictor Correlation Summary:\n")
            rs = [res.get('r', 0) for res in predictor_results]
            if rs:
                avg_abs = np.mean([abs(r) for r in rs])
                num_pos = sum(1 for r in rs if r > 0)
                num_neg = sum(1 for r in rs if r < 0)
                sorted_by_r = sorted(predictor_results, key=lambda x: x.get('r', 0))
                lowest  = sorted_by_r[0]
                highest = sorted_by_r[-1]

                f.write(f"- Average |r|: {avg_abs:.3f}\n")
                f.write(f"- Positive correlations: {num_pos}/{len(rs)}\n")
                f.write(f"- Negative correlations: {num_neg}/{len(rs)}\n")
                f.write(f"- Highest r: {highest.get('label','Unknown')} (r = {highest.get('r'):+.3f}, p = {highest.get('p'):.4g})\n")
                f.write(f"- Lowest  r: {lowest.get('label','Unknown')} (r = {lowest.get('r'):+.3f}, p = {lowest.get('p'):.4g})\n")

            # 4) feature retention recommendations
            f.write("\nFeature Retention Recommendations:\n")
            f.write("-" * 60 + "\n")
            f.write(f"Predictors sorted by |r| (threshold = {sig_corr_threshold}):\n\n")
            sorted_preds = sorted(predictor_results, key=lambda x: abs(x.get('r', 0)), reverse=True)
            for pred in sorted_preds:
                lbl   = pred.get('label', 'Unknown')
                r_val = pred.get('r', 0)
                mark  = "✓ KEEP" if abs(r_val) >= sig_corr_threshold else "  drop"
                f.write(f"{mark}  {lbl:<30} (r = {r_val:+.3f})\n")
            f.write("\nRecommendations:\n")
            f.write("  • Keep all predictors marked ✓ KEEP when building your next PCA or feature set.\n")
            f.write("  • Remove (or review) those marked “drop” to reduce noise.\n")

            # footer
            f.write("\n" + "=" * 60 + "\n")

        print(f"[{ts}] [REPORT] Significance report saved to '{output_filename}'")
        return True

    except Exception as report_err:
        print(f"[{ts}] [REPORT] WARNING: Could not write report '{output_filename}': {report_err}")
        traceback.print_exc()
        return False
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#model training with statistical calc.
def train_model():
    global scaler, pca, q_values, k_values, s_values, r_values, r_char, S2_BA, USE_PCA, FEATURE_COLUMNS, last_training_stats

    print(f"\n--- Training Model ---");
    print(f"Using features: {FEATURE_COLUMNS}")
    print(f"PCA Enabled: {USE_PCA}")

    scaler = None
    pca = None
    q_values = []
    k_values = []
    s_values = []
    r_values = []
    r_char = 0
    S2_BA = 0
    last_training_stats.clear()

    if not FEATURE_COLUMNS:
        print("[TRAIN ERROR] No features defined.")
        return False

    try:
        df_dataset = load_dataset()
        required_cols = FEATURE_COLUMNS + [AGE_COLUMN_NAME]
        if not all(col in df_dataset.columns for col in required_cols):
            print(f"[TRAIN ERROR] Dataset missing cols."); return False
        if df_dataset.empty:
            print("[TRAIN ERROR] Dataset empty."); return False

        df_analysis = df_dataset.dropna(subset=required_cols).copy()
        if len(df_analysis) < 2:
            print(f"[TRAIN ERROR] Need >= 2 complete rows, found {len(df_analysis)}."); return False

        print(f"Training model on {len(df_analysis)} complete rows.")

        #standardize
        scaler = StandardScaler()
        df_standardized = scaler.fit_transform(df_analysis[FEATURE_COLUMNS])
        X_age = df_analysis[[AGE_COLUMN_NAME]].values

        #choose regressors (PCA or raw)
        data_for_regression = None
        column_labels_for_regression = []
        num_regressors = len(FEATURE_COLUMNS)

        if USE_PCA:
            n_components = min(len(FEATURE_COLUMNS), len(df_analysis))
            if n_components <= 0:
                print("[TRAIN ERROR] PCA components <= 0."); return False
            pca = PCA(n_components=n_components)
            principal_components = pca.fit_transform(df_standardized)
            data_for_regression = principal_components
            column_labels_for_regression = [f'PC{i+1}' for i in range(n_components)]
            num_regressors = n_components
        else:
            data_for_regression = df_standardized
            column_labels_for_regression = FEATURE_COLUMNS

        #regression loop, collect q, k, s, r, p
        temp_q, temp_k, temp_s, temp_r, temp_p = [], [], [], [], []
        for i in range(num_regressors):
            y_predic = data_for_regression[:, i].reshape(-1, 1)
            lbl = column_labels_for_regression[i]
            if np.var(y_predic) < 1e-10:
                slope, intercept, rmse, r_j, p_j = 0.0, 0.0, 0.0, 0.0, 1.0
            else:
                try:
                    mdl = LinearRegression().fit(X_age, y_predic)
                    slope    = mdl.coef_[0][0]
                    intercept= mdl.intercept_[0]
                    y_hat    = mdl.predict(X_age)
                    rmse     = np.sqrt(mean_squared_error(y_predic, y_hat))
                    if X_age.shape[0] > 1 and np.std(X_age.flatten()) > 1e-9 and np.std(y_predic.flatten()) > 1e-9:
                        r_j, p_j = pearsonr(X_age.flatten(), y_predic.flatten()); print(r_j)
                    else:
                        r_j, p_j = 0.0, 1.0
                except Exception as reg_e:
                    print(f"[TRAIN ERROR] Regression failed: {reg_e}.")
                    slope, intercept, rmse, r_j, p_j = 0.0, 0.0, 0.0, 0.0, 1.0

            temp_q.append(intercept)
            temp_k.append(slope)
            temp_s.append(rmse if not np.isnan(rmse) else 0.0)
            temp_r.append(r_j if not np.isnan(r_j) else 0.0)
            temp_p.append(p_j if not np.isnan(p_j) else 1.0)

        #assign global lists
        q_values[:] = temp_q[:num_regressors]
        k_values[:] = temp_k[:num_regressors]
        s_values[:] = temp_s[:num_regressors]
        r_values[:] = temp_r[:num_regressors]

        #KDM function, BA_E, S2_BA, BA_EC
        r_char = calculate_r_char(r_values)
        print(f"Calculated r_char: {r_char:.4f}")

        ca_list = df_analysis[AGE_COLUMN_NAME].tolist()
        ba_e_list = []
        df_results_dict = {AGE_COLUMN_NAME: ca_list, 'BA_E': [], 'BA_EC': []}
        for idx in range(len(df_analysis)):
            iv = data_for_regression[idx, :].tolist()
            ba_e = calculate_ba_e(iv, q_values, k_values, s_values)
            ba_e_list.append(ba_e)
            df_results_dict['BA_E'].append(ba_e)

        m = len(FEATURE_COLUMNS)
        S2_BA = calculate_s2_ba(ba_e_list, ca_list, r_char, m)
        print(f"Calculated S2_BA: {S2_BA}")
        if S2_BA is None or np.isnan(S2_BA):
            print("Warning: S2_BA is NaN."); S2_BA = np.nan

        ba_ec_list = []
        for idx in range(len(df_analysis)):
            iv = data_for_regression[idx, :].tolist()
            ca_val = ca_list[idx]
            ba_ec = np.nan if np.isnan(S2_BA) or np.isnan(ca_val) else calculate_ba_ec(iv, q_values, k_values, s_values, ca_val, S2_BA)
            ba_ec_list.append(ba_ec)
            df_results_dict['BA_EC'].append(ba_ec)

        df_results = pd.DataFrame(df_results_dict)
        df_results['BA_Accel'] = df_results['BA_EC'] - df_results[AGE_COLUMN_NAME]

        #training statistics
        stat_cols = [AGE_COLUMN_NAME, 'BA_EC', 'BA_Accel']
        valid = df_results.dropna(subset=stat_cols).copy()
        if not valid.empty and len(valid) > 1:
            print(f"Calculating statistics on {len(valid)} valid rows...")
            try:
                last_training_stats['MAE_BAEC_vs_CA'] = mean_absolute_error(valid[AGE_COLUMN_NAME], valid['BA_EC'])
                last_training_stats['Corr_BAEC_vs_CA'] = np.corrcoef(valid[AGE_COLUMN_NAME], valid['BA_EC'])[0,1]
                last_training_stats['Mean_BA_Accel'] = valid['BA_Accel'].mean()
                last_training_stats['SD_BA_Accel']   = valid['BA_Accel'].std()
                last_training_stats['Mean_BA_EC']    = valid['BA_EC'].mean()
                last_training_stats['SD_BA_EC']      = valid['BA_EC'].std()
                print(f"Calculated Stats: MAE={last_training_stats.get('MAE_BAEC_vs_CA', 'N/A'):.2f}, Corr={last_training_stats.get('Corr_BAEC_vs_CA', 'N/A'):.2f}")
            except Exception as stat_err:
                print(f"[TRAIN ERROR] Could not calculate statistics: {stat_err}")
                last_training_stats.clear()
                last_training_stats['Stats_Error'] = str(stat_err)
        else:
            print(f"Warning: Not enough valid data points ({len(valid)}) to calculate reliable training statistics.")
            last_training_stats.clear()

        #save and plot
        save_intermediate_results(df_results)
        if SAVE_PLOTS_ON_TRAIN:
            generate_and_save_plots(df_results, PLOTS_OUTPUT_DIR)

        #significance report
        predictor_type_desc = "PCA components" if USE_PCA else "Standardized features"
        predictor_results = [
            {'label': column_labels_for_regression[i], 'r': r_values[i], 'p': temp_p[i]}
            for i in range(num_regressors)
        ]
        write_significance_report(
            predictor_results,
            num_analysis_rows=len(df_analysis),
            num_predictors_analyzed=num_regressors,
            predictor_type_desc=predictor_type_desc,
            output_filename="BioDyn-significance-report.txt"
        )

        print("--- Model Training Completed Successfully ---")
        return True

    except Exception as e:
        print(f"ERROR during model training: {e}")
        traceback.print_exc()
        #reset globals
        pca = None; q_values=[]; k_values=[]; s_values=[]; r_values=[]; r_char=0; S2_BA=0
        last_training_stats.clear()
        return False
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#here we calculate biological age
def calculate_user_biological_age(user_data):
    
    global scaler, pca, q_values, k_values, s_values, S2_BA, USE_PCA, FEATURE_COLUMNS
    #print("\n[BA CALC DEBUG] Inside calculate_user_biological_age")
    
    if not FEATURE_COLUMNS: 
        flash("Config error...", 'danger')
        return None
    
    if scaler is None: 
        flash("Model (Scaler) not trained...", 'danger')
        return None
    
    if USE_PCA and pca is None: 
        flash("Model (PCA) not trained...", 'danger')
        return None
    
    age_col_key = 'Age' 
    required_input_keys = FEATURE_COLUMNS + [age_col_key]
    
    if age_col_key not in user_data: 
        flash(f"Error: {age_col_key} missing.", 'danger')
        return None
    
    missing_keys = [key for key in required_input_keys if key not in user_data]
    
    if missing_keys: 
        flash(f"Error: Missing: {', '.join(missing_keys)}.", 'danger')
        return None
    
    try: 
        df_input = pd.DataFrame([user_data])[FEATURE_COLUMNS]
    
    except KeyError as e: 
        flash(f"Error: Input access: {e}.", 'danger')
        return None
    
    for col in FEATURE_COLUMNS:
        if col in df_input.columns:

            try: df_input[col] = pd.to_numeric(df_input[col])
            
            except (ValueError, TypeError): 
                flash(f"Error: Invalid input '{col}'.", 'danger')
                #print(f"[BA CALC DEBUG] ValueError converting '{col}'")
                return None
        else: 
            flash(f"Error: Feature '{col}' missing.", 'danger') 
            return None
    
    try: 
        chronological_age = float(user_data[age_col_key])
    
    except (ValueError, TypeError): 
        flash(f"Error: Invalid {age_col_key}.", 'danger')
        return None
    
    try:
        df_input_standardized = scaler.transform(df_input)
        
        values_for_ba_calc = pca.transform(df_input_standardized)[0] if USE_PCA else df_input_standardized[0]
        
        num_predictors = len(values_for_ba_calc)
        
        if len(q_values) < num_predictors: 
            flash("Error: Coeff mismatch.", "danger")
            return None
        

        #print(f"[BA CALC DEBUG] --> Calling calculate_ba_ec with:")
        #print(f"[BA CALC DEBUG]     x_values (first 5): {values_for_ba_calc[:5]}")
        #print(f"[BA CALC DEBUG]     q_values (first 5): {q_values[:num_predictors][:5]}")
        #print(f"[BA CALC DEBUG]     k_values (first 5): {k_values[:num_predictors][:5]}")
        #print(f"[BA CALC DEBUG]     s_values (first 5): {s_values[:num_predictors][:5]}")
        #print(f"[BA CALC DEBUG]     ca: {chronological_age}")
        #print(f"[BA CALC DEBUG]     s2_ba: {S2_BA}")

        
        biological_age = calculate_ba_ec(values_for_ba_calc, q_values[:num_predictors], k_values[:num_predictors], s_values[:num_predictors], chronological_age, S2_BA)
        
        if biological_age is None or np.isnan(biological_age): 
            flash("Error: Could not calculate BA.", 'warning')
            return None
        
        return biological_age
    
    except Exception as e: 
        
        flash(f"Error processing input: {e}", 'danger')
        
        #print(f"Error transforming/calculating user BA: {e}")
        
        traceback.print_exc()
        return None
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#flask route here -- standard flask stuff
@app.route('/', methods=['GET', 'POST'])
def index():
    
    biological_age_result = None
    age_col_form = 'Age'
    
    if request.method == 'POST':
        
        form_data = {}; user_data_display = {}
        try:
            for feature in FEATURE_COLUMNS: 
                
                form_data[feature] = request.form[feature]
                
                user_data_display[feature] = request.form[feature]

            form_data[age_col_form] = request.form[age_col_form]
            
            user_data_display[age_col_form] = request.form[age_col_form]
            
            user_data_float = {}
            
            for key, value in form_data.items():
                
                try: 
                    user_data_float[key] = float(value)
                
                except (ValueError, TypeError): 
                    
                    flash(f"Invalid input: '{value}' for {key}.", 'danger')
                    
                    return render_template('index.html', biological_age=None, user_data=user_data_display, feature_columns=FEATURE_COLUMNS)

            model_ready = scaler is not None and (pca is not None if USE_PCA else True) and k_values

            if not model_ready:
                flash("Model not ready. Initializing...", 'warning')

                print("Model not ready. Running train_model.")
                
                training_success = train_model()
                
                if not training_success: 
                    flash("Failed init model.", 'danger') 
                    return render_template('index.html', biological_age=None, user_data=user_data_display, feature_columns=FEATURE_COLUMNS)
                
                model_ready = scaler is not None and (pca is not None if USE_PCA else True) and k_values
                
                if not model_ready: 
                    flash("Model init failed.", 'danger'); 
                    return render_template('index.html', biological_age=None, user_data=user_data_display, feature_columns=FEATURE_COLUMNS)
            
            biological_age_result = calculate_user_biological_age(user_data_float)
            
            if biological_age_result is not None: 
                flash(f"Biological Age (BA_EC): {biological_age_result:.2f}", 'success')
            
            print(f"[DBG] Value passed to template: biological_age = {biological_age_result}") # <-- ADD THIS LINE
        
        except KeyError as e: 
            missing_field = str(e).strip("'"); 
            flash(f"Missing field: {missing_field}.", 'danger')
            print(f"KeyError: {e}")
        
        except Exception as e: 
            flash(f"Unexpected error: {e}", 'danger') 
            print(f"Unexpected error index POST: {e}")
            traceback.print_exc()
        
        return render_template('index.html', biological_age=biological_age_result, user_data=user_data_display, feature_columns=FEATURE_COLUMNS)
    else: #get Request
        user_data_display = {}
        
        for feature in FEATURE_COLUMNS: 
            user_data_display[feature] = DEFAULT_BIOMARKER_VALUES.get(feature, '')
        
        user_data_display[age_col_form] = DEFAULT_BIOMARKER_VALUES.get(age_col_form, DEFAULT_AGE_FORM_VALUE)
        
        #return render_template('index.html', biological_age=None, user_data=user_data_display, feature_columns=FEATURES_COLUMNS)
        return render_template('index.html', biological_age=None, user_data=user_data_display, feature_columns=FEATURE_COLUMNS)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#this handles the uploads when merging new datasets
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files: 
            flash("No file part...", 'danger')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '': 
            flash("No selected file...", 'danger')
            return redirect(request.url)
        
        if file and file.filename.lower().endswith(('.csv', '.xlsx', '.xls')):
            try:
                if file.filename.lower().endswith('.csv'): 
                    df_uploaded = pd.read_csv(file)
                
                else: df_uploaded = pd.read_excel(file)
                
                df_uploaded.columns = df_uploaded.columns.str.strip()
                
                print("Uploaded columns:", df_uploaded.columns.tolist())
                
                if AGE_COLUMN_NAME not in df_uploaded.columns: 
                    flash(f"Uploaded file must contain age column: '{AGE_COLUMN_NAME}'", 'danger')
                    return redirect(request.url)
                
                merge_with_existing_data(df_uploaded, DATASET_PATH); 
                
                print("Merged uploaded data.")
                
                print("Retraining model..."); training_success = train_model()
                
                if training_success: flash("File uploaded, merged, and model retrained!", "success")
                
                else: flash("Upload complete, but model retraining failed.", "warning")
                
                return redirect(url_for('index'))
            except Exception as e: 
                
                flash(f"Error processing upload: {e}", 'danger')
                print(f"Exception upload POST: {e}")
                traceback.print_exc()
                return redirect(request.url)

        else: flash("Please upload valid .csv or Excel.", 'danger')
        
        return redirect(request.url)
    
    return render_template('upload.html')
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#this section is for web plot function
@app.route('/plot')
def plot():
    
    print("\n[PLOT WEB] Accessed /plot route (Requesting BA_EC)")
    
    model_ready = (
        scaler is not None and
        (pca is not None if USE_PCA else True) and
        k_values and #check if lists are populated
        S2_BA is not None and not np.isnan(S2_BA) #and S2_BA > 1e-9 #check S2_BA validity
    )
    
    if not model_ready:
        
        print("[PLOT WEB] Model not ready or S2_BA invalid for BA_EC plot.")
        
        flash("Model not ready or key parameters (like S2_BA) invalid. Cannot generate BA_EC plot.", 'warning')
        
        return redirect(url_for('index'))

    df_dataset = load_dataset()

    if df_dataset.empty:
        print("[PLOT WEB] Dataset empty.")
        
        flash("Dataset empty.", 'danger')
        
        return redirect(url_for('index'))

    #filter data for required columns
    df_plot_data = df_dataset.dropna(subset=FEATURE_COLUMNS + [AGE_COLUMN_NAME]).copy()
    
    if df_plot_data.empty:
        
        print("[PLOT WEB] No complete rows found.")
        
        flash("No complete data rows for plotting.", 'warning')
        
        return redirect(url_for('index'))

    try:
        #transform features
        print("[PLOT WEB] Transforming data...")
        
        plot_features_std = scaler.transform(df_plot_data[FEATURE_COLUMNS])
        
        values_for_ba_plot = pca.transform(plot_features_std) if USE_PCA else plot_features_std
        
        num_predictors = values_for_ba_plot.shape[1]
        
        print(f"[PLOT WEB] Data transformed. Shape: {values_for_ba_plot.shape}. Num predictors: {num_predictors}")

        #check coefficients length
        if len(q_values) < num_predictors or len(k_values) < num_predictors or len(s_values) < num_predictors:
             print("[PLOT WEB] Coeff length mismatch.")
             flash("Plot Error: Coefficient mismatch.", "danger")
             return redirect(url_for('index'))

        #calculate BA_EC for each plot point
        
        print("[PLOT WEB] Calculating BA_EC...")
        
        #get the corresponding chronological ages
        ca_list_plot = df_plot_data[AGE_COLUMN_NAME].tolist()
        
        calculated_bas_ec = []

        for i, data_row in enumerate(values_for_ba_plot):
            ca = ca_list_plot[i] #get the chronological age for this row
            if np.isnan(ca): #skip if CA is NaN
                 ba_ec = np.nan
            else:
                 ba_ec = calculate_ba_ec(
                     data_row,
                     q_values[:num_predictors],
                     k_values[:num_predictors],
                     s_values[:num_predictors],
                     ca,       #pass chronological age
                     S2_BA     #pass global S2_BA
                 )
            calculated_bas_ec.append(ba_ec)

        df_plot_data['BA_EC_Calculated'] = calculated_bas_ec
        
        print(f"[PLOT WEB] Finished BA_EC calc. Example: {calculated_bas_ec[:5]}")
        
        df_plot_data.dropna(subset=['BA_EC_Calculated'], inplace=True) #drop rows where BA_EC failed
        
        print(f"[PLOT WEB] df_plot_data shape after dropping NaN BA_EC: {df_plot_data.shape}")

        if df_plot_data.empty:
            print("[PLOT WEB] No valid BA_EC values calculated.")
            flash("Could not calculate Biological Age (BA_EC) for any data points.", "warning")
            return redirect(url_for('index'))

        #generate Plot (using BA_EC)
        print(f"[PLOT WEB] Generating web plot with {len(df_plot_data)} points (BA_EC).")
        plt.figure(figsize=(10, 8))

        #plot existing vs uploaded using BA_EC_Calculated
        existing_data = df_plot_data[df_plot_data['source'] == 'existing']
        if not existing_data.empty:
            plt.scatter(existing_data[AGE_COLUMN_NAME], existing_data['BA_EC_Calculated'], #Y-axis uses BA_EC
                        label=f'Existing (n={len(existing_data)})', alpha=0.6, color='blue', s=50)
        
        uploaded_data = df_plot_data[df_plot_data['source'] == 'uploaded']
        if not uploaded_data.empty:
            plt.scatter(uploaded_data[AGE_COLUMN_NAME], uploaded_data['BA_EC_Calculated'], #Y-axis uses BA_EC
                        label=f'Uploaded (n={len(uploaded_data)})', alpha=0.8, color='red', s=70, marker='^')

        #plot y=x line using BA_EC_Calculated for y-limits
        min_val = np.nanmin([df_plot_data[AGE_COLUMN_NAME].min(), df_plot_data['BA_EC_Calculated'].min()])
        
        max_val = np.nanmax([df_plot_data[AGE_COLUMN_NAME].max(), df_plot_data['BA_EC_Calculated'].max()])
        
        if pd.notna(min_val) and pd.notna(max_val) and min_val < max_val:
            plt.plot([min_val, max_val], [min_val, max_val], color='grey', linestyle='--', label='CA = BA_EC') #label updated

        #update labels and title
        plt.xlabel(f'Chronological Age ({AGE_COLUMN_NAME})')
        
        plt.ylabel('Calculated Biological Age (BA_EC, Years)')
        
        plt.title(f'Web Plot: CA vs BA_EC (PCA {"Enabled" if USE_PCA else "Disabled"})')
        
        plt.legend(); plt.grid(True, linestyle='--', alpha=0.6); plt.tight_layout()

        #send plot image
        img_io = io.BytesIO()
        
        plt.savefig(img_io, format='png', bbox_inches='tight')
        
        img_io.seek(0)
        
        plt.close()
        
        print("[PLOT WEB] Sending file.")
        
        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        print(f"[PLOT WEB] Error during plot generation: {e}")
        flash(f"Error generating plot: {e}", 'danger')
        import traceback; traceback.print_exc()
        return redirect(url_for('index'))
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#init biodyn
def initialize_app():
    
    print("\n=== BioDyn Initializing ===")
    
    if not FEATURE_COLUMNS: print("*"*60+"\nINIT ERROR: BIOMARKER_FEATURES empty.\n"+"*"*60); return
    print(f"Using features: {FEATURE_COLUMNS}"); print(f"Age column: {AGE_COLUMN_NAME}"); print(f"PCA Enabled: {USE_PCA}")
    if SAVE_PLOTS_ON_TRAIN: # Use global var
        try: os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True); print(f"Plots directory '{PLOTS_OUTPUT_DIR}' ensured.") # Use global var
        except OSError as e: print(f"Warning: Could not create plots directory '{PLOTS_OUTPUT_DIR}': {e}")
    loaded_ok = load_intermediate_results()
    if loaded_ok is None:
        print("Initialization: Results not loaded/incompatible. Training new model...")
        training_success = train_model()
        if training_success: print("Initialization: New model trained successfully.")
        else: print("*"*60+"\nINIT WARNING: Failed train on startup.\n"+"*"*60)
    else:
        print("Initialization: Compatible results loaded. Re-fitting model components/results...")
        training_success = train_model()
        if training_success: print("Initialization: Model components fitted/results updated.")
        else: print("*"*60+"\nINIT WARNING: Loaded coeffs but FAILED refit/update.\n"+"*"*60)

    
    print("=== BioDyn Initialization Complete ===")
    
    print("\n")
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#app run
initialize_app()
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0')