# BioDyn: Biological Age Calculator 🧬⏱️

**Developed by Mehrdad S. Beni - April 2025**

> A Flask web application to estimate Biological Age (BA) using biomarker data, based on the principles of the Klemera and Doubal Method (KDM). BioDyn provides an interactive interface for BA calculation, data management (uploading & merging), model retraining, and visualization.

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.x-orange.svg)](https://flask.palletsprojects.com/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-used-brightgreen.svg)](https://scikit-learn.org/)

## ✨ Features

* **Web Interface:** User-friendly form to input biomarker values and chronological age for BA calculation.
* **Biological Age Estimation:** Calculates BA_EC (Chronological Age corrected Biological Age) using the KDM framework.
* **KDM Parameter Calculation:** Determines key KDM parameters like `r_char` (characteristic correlation) and `S2_BA` (biological age variance).
* **Data Upload & Merging:** Upload new datasets (CSV/Excel) which are automatically merged with the existing data.
* **Automatic Model Retraining:** The underlying statistical model (regressions, KDM parameters) is automatically retrained after data merging.
* **Optional PCA:** Supports using Principal Component Analysis (PCA) on biomarker features before regression (configurable).
* **Intermediate Results:** Saves and loads model parameters (`scaler`, `pca`, coefficients, `S2_BA`, `r_char`) for persistence.
* **Data Visualization:**
    * Generates plots comparing Chronological Age (CA) vs. Biological Age (BA_EC).
    * Generates plots showing Biological Age Acceleration (BA_EC - CA).
    * Generates histograms/density plots of BA Acceleration distribution.
    * Provides an interactive web plot (`/plot`) showing CA vs BA_EC for the current dataset, distinguishing between original and uploaded data.
* **Configurable:** Easily configure dataset paths, biomarker features, PCA usage, plot labels, and default values via `config.yaml`.

---

## 🚀 Getting Started

### Prerequisites

* Python (version 3.10 or higher recommended)
* `pip` (Python package installer)

### Installation & Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/msbCyricTohoku/BioDyn.git
    cd BioDyn
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Optional: It's highly recommended to use a virtual environment)*
    ```bash
    # python -m venv venv
    # source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate  # On Windows
    # pip install -r requirements.txt
    ```

3.  **Prepare `config.yaml`:**
    * Ensure the `config.yaml` file is present in the project root directory.
    * **Crucially, edit `config.yaml`:**
        * Set `DATASET_PATH` to the name of your primary data file (e.g., `basedata.csv`).
        * Set `AGE_COLUMN_NAME` to the exact header name of the age column in your CSV.
        * List the **exact** header names of the biomarker columns you want to use under `BIOMARKER_FEATURES`. **These must match your CSV headers precisely.**
        * Configure `USE_PCA` (`true` or `false`).
        * Adjust other paths and settings as needed.

4.  **Prepare Initial Dataset:**
    * Make sure the data file specified in `DATASET_PATH` (e.g., `basedata.csv`) exists and contains the necessary columns (Age and the features listed in `BIOMARKER_FEATURES`).

5.  **Run the Application:**
    ```bash
    python app.py
    ```
    The application should start, initialize the model (potentially training it if intermediate results aren't found or are incompatible), and be accessible at `http://127.0.0.1:5000`.

---

## ⚙️ Configuration (`config.yaml`)

The application's behavior is controlled by `config.yaml`. Key settings include:

* `DATASET_PATH`: Path to the main CSV data file.
* `INTERMEDIATE_RESULTS_PATH`: Path to the Excel file where trained model parameters are saved/loaded.
* `AGE_COLUMN_NAME`: Exact name of the chronological age column in your data.
* `BIOMARKER_FEATURES`: A list of exact column names from your data file to be used as biomarkers in the model. **The order might matter if `USE_PCA` is false, but primarily the names must match exactly.**
* `USE_PCA`: Set to `true` to enable PCA, `false` otherwise.
* `DEFAULT_BIOMARKER_VALUES`: Default values to pre-fill the input form on the main page. Keys must match `BIOMARKER_FEATURES`.
* `DEFAULT_AGE_FORM_VALUE`: Default value for the age input field.
* `PLOTS_OUTPUT_DIR`: Directory where plots generated during training are saved.
* `SAVE_TRAINING_BA_VALUES`: `true` to save calculated BA_E/BA_EC for the training set in the intermediate results file.
* `SAVE_PLOTS_ON_TRAIN`: `true` to automatically save plots when the model is trained/retrained.
* `PLOT_FILENAME_TIMESTAMP`: `true` to add a timestamp to saved plot filenames.
* `plot_labels`: Customize titles and axis labels for the generated plots.

---

## 💻 Usage

1.  **Main Page (`/`):**
    * Enter the chronological age and values for the configured biomarkers in the form.
    * Click "Calculate Biological Age".
    * The estimated BA_EC will be displayed (flashed message), and the form will retain your input.
2.  **Upload Data (`/upload`):**
    * Navigate to the `/upload` page.
    * Choose a CSV or Excel file containing new data. The file **must** include the configured `AGE_COLUMN_NAME` and ideally all `BIOMARKER_FEATURES` for the data to be fully utilized in retraining.
    * Click "Upload".
    * The data will be merged with the existing dataset (`DATASET_PATH`), and the model will be automatically retrained. You will be redirected to the main page with a status message.
3.  **View Plot (`/plot`):**
    * Navigate to the `/plot` page.
    * A scatter plot comparing Chronological Age vs. calculated Biological Age (BA_EC) for all *complete* data points in the current dataset will be displayed. Points from the original dataset and uploaded datasets are marked differently.

---

## 🛠️ How It Works (High Level)

1.  **Data Loading:** Reads data from the CSV file defined in `config.yaml`.
2.  **Preprocessing:**
    * Filters data to include only rows with complete biomarker and age information.
    * Standardizes biomarker features (using `StandardScaler`).
    * Applies PCA if `USE_PCA` is `true`.
3.  **Regression:** Performs linear regression for each feature (or principal component) against chronological age to find slopes (`k_values`), intercepts (`q_values`), and standard errors (`s_values`). Calculates correlations (`r_values`).
4.  **KDM Parameter Calculation:** Computes `r_char` and `S2_BA` based on the regression results.
5.  **BA_EC Calculation:** Uses the formula derived from KDM, incorporating the biomarker values, regression coefficients (`q`, `k`, `s`), chronological age (`ca`), and `S2_BA`.
6.  **Web Interface:** Flask handles routing, form processing, calls calculation functions, and renders HTML templates.
7.  **Persistence:** Saves/loads the scaler, PCA object (if used), and calculated coefficients/parameters to/from an Excel file (`INTERMEDIATE_RESULTS_PATH`).

---

