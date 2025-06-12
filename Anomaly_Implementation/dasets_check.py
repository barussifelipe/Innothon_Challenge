import os
import pandas as pd


#Paths
PIVOTED_DATA_DIR = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/Study_datasets/Consumption/new_pivoted_data'


def validate_pivoted_data(pivoted_data_dir, max_plausible_consumption: float = None, min_std_dev_threshold: float = 1e-6):
    """
    Performs enhanced purity checks on the pivoted consumption data files.

    Args:
        pivoted_data_dir (str): Directory containing the pivoted SUPPLYXXX.csv files.
        max_plausible_consumption (float, optional): A threshold for physically impossible
                                                     high consumption values. If None, this check is skipped.
        min_std_dev_threshold (float): Threshold for standard deviation to detect near-constant data.
                                       If a column's std dev is below this, it's considered near-constant.
    """
    print("\n--- Starting Post-Pivoting Data Validation ---")
    validation_issues = {}

    supply_files = [f for f in os.listdir(pivoted_data_dir) if f.endswith('.csv')]
    
    if not supply_files:
        print(f"No CSV files found in {pivoted_data_dir}. Please check the path and if files are generated.")
        return

    for filename in supply_files:
        supply_id = filename.replace('.csv', '')
        filepath = os.path.join(pivoted_data_dir, filename)
        
        try:
            # Read with date as index and specify decimal separator
            df = pd.read_csv(filepath, index_col='date', decimal=',') 
            
            # Convert index to datetime for date checks
            df.index = pd.to_datetime(df.index)

            # Ensure all consumption columns are numeric and fill any resulting NaNs with 0
            consumption_cols = df.columns
            for col in consumption_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df.fillna(0, inplace=True)
            
            # --- Purity Checks ---

            # 1. Check for Negative Consumption
            if (df < 0).any().any():
                issue = f"Contains negative consumption values."
                validation_issues.setdefault(supply_id, []).append(issue)

            # 2. Check for Constant/Zero-Variance Consumption (Existing Check)
            if df.nunique().sum() == 1 and df.iloc[0, 0] == 0: # All values are 0
                issue = f"All consumption values are zero."
                validation_issues.setdefault(supply_id, []).append(issue)
            elif df.nunique().sum() == df.shape[1] and (df.std() == 0).all(): # All columns have zero std (constant)
                 issue = f"Contains constant consumption values (zero variance)."
                 validation_issues.setdefault(supply_id, []).append(issue)

            # --- NEW CHECK: Check for Near-Constant / Extremely Low Variance ---
            # This catches supplies that are not exactly zero variance, but very close
            if not df.empty and (df.std().max() < min_std_dev_threshold):
                issue = f"Contains near-constant consumption values (max std dev: {df.std().max():.2e})."
                validation_issues.setdefault(supply_id, []).append(issue)


            # 3. Check for Duplicate Dates (already handled by pivot, but good to check)
            if df.index.duplicated().any():
                issue = f"Contains duplicate dates in index."
                validation_issues.setdefault(supply_id, []).append(issue)

            # 4. Check for Correct Number of Columns (already handled by reindex in preprocessing, but good to check)
            if df.shape[1] != 96:
                issue = f"Incorrect number of quarter-hour columns ({df.shape[1]} instead of 96)."
                validation_issues.setdefault(supply_id, []).append(issue)

            # 5. Check for Extreme Outliers (High Values)
            if max_plausible_consumption is not None:
                if (df > max_plausible_consumption).any().any():
                    max_val = df.max().max()
                    issue = f"Contains extremely high consumption values (max: {max_val:.2f}) exceeding plausible limit ({max_plausible_consumption})."
                    validation_issues.setdefault(supply_id, []).append(issue)

            # 6. Check for Large Gaps in Dates (Missing Days)
            if not df.empty:
                expected_dates = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
                missing_dates = expected_dates.difference(df.index)
                if not missing_dates.empty:
                    issue = f"Missing {len(missing_dates)} days in date range. Examples: {missing_dates[:5].strftime('%Y-%m-%d').tolist()}."
                    validation_issues.setdefault(supply_id, []).append(issue)

        except Exception as e:
            print(f"  Error validating {filename}: {e}")
            validation_issues.setdefault(supply_id, []).append(f"Validation error: {e}")
            continue

    print("\n--- Validation Complete ---")
    if validation_issues:
        print("Summary of Issues Found:")
        for sid, issues in validation_issues.items():
            print(f"  {sid}: {'; '.join(issues)}")
    else:
        print("No major data purity issues detected in pivoted data.")


validate_pivoted_data(PIVOTED_DATA_DIR, 4)
