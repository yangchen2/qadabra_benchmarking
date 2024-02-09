import pandas as pd
import biom
from biom import load_table
from biom.util import biom_open
import qiime2 as q2
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import logging

### This script reads in the original 16S and metagenomic BIOM tables and does the subsetting for subsequent analyses

# Setup logging
logging.basicConfig(filename='../logs/subset_tables.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def read_and_convert_biom(biom_16S_path: str, biom_metagenomic_path: str):
    """
    Read two BIOM-format tables and convert them into Pandas dataframes.

    Parameters:
    biom_16S (str): The file path to the 16S biom table.
    biom_metagenomic (str): The file path to the metagenomic biom table.

    Returns:
    tuple: A tuple containing two dataframes:
           - The first dataframe corresponds to the 16S biom table.
           - The second dataframe corresponds to the metagenomic biom table.
    """

    logging.info(f"STEP 1: Loading BIOM files: {biom_16S_path} and {biom_metagenomic_path}")

    try:
        # Load BIOM tables
        biom_table_16S = load_table(biom_16S_path)
        biom_metagenomic = load_table(biom_metagenomic_path)

        # Convert BIOM tables to pandas DataFrame
        df_16S = pd.DataFrame(biom_table_16S.to_dataframe().transpose())
        df_metagenomic = pd.DataFrame(biom_metagenomic.to_dataframe().transpose())

        # Log shape of DataFrames
        logging.info("16S table shape: " + str(df_16S.shape))
        logging.info("Metagenomic table shape: " + str(df_metagenomic.shape))

        logging.info("-----> COMPLETED: BIOM files successfully converted to DataFrames")
        return df_16S, df_metagenomic
    except Exception as e:
        logging.error(f"-----> ERROR in processing BIOM files: {e}")
        raise


def remove_set_differences(df_16S: pd.DataFrame, df_metagenomic: pd.DataFrame, in_place: bool = False) -> tuple:
    """
    Identify and remove sample IDs that are unique to either the 16S table or the metagenomic table and returns filtered tables.

    Parameters:
    df_16S (pd.DataFrame): Input Pandas DataFrame of the 16S table.
    df_metagenomic (pd.DataFrame): Input Pandas DataFrame of the metagenomic table.
    in_place (bool): If True, modifies the dataframes in-place. Defaults to False.

    Returns:
    tuple: A tuple containing two processed DataFrames.
    """
    logging.info("STEP 2: Starting to align sample IDs between the 16S and metagenomic tables.")

    if df_16S.empty or df_metagenomic.empty:
        logging.warning("One or both input DataFrames are empty.")
        return df_16S, df_metagenomic

    try:
        # Identify unique indexes
        indexes_only_in_16S = df_16S.index[~df_16S.index.isin(df_metagenomic.index)]
        indexes_only_in_metagenomic = df_metagenomic.index[~df_metagenomic.index.isin(df_16S.index)]

        logging.info(f"-----> Indexes only in 16S: {indexes_only_in_16S}")
        logging.info(f"-----> Indexes only in metagenomic: {indexes_only_in_metagenomic}")

        # Remove the unique indexes
        if not in_place:
            df_16S, df_metagenomic = df_16S.copy(), df_metagenomic.copy()

        df_16S.drop(indexes_only_in_16S, inplace=True)
        df_metagenomic.drop(indexes_only_in_metagenomic, inplace=True)

        # Check if done correctly
        if df_16S.index.equals(df_metagenomic.index):
            # Log shape of DataFrames
            logging.info("16S table shape: " + str(df_16S.shape))
            logging.info("Metagenomic table shape: " + str(df_metagenomic.shape))
            logging.info("-----> COMPLETED: DataFrame samples successfully aligned.")
        else:
            # Log shape of DataFrames
            logging.info("16S table shape: " + str(df_16S.shape))
            logging.info("Metagenomic table shape: " + str(df_metagenomic.shape))
            logging.error("-----> ERROR: DataFrames still have different samples after processing.")

        return df_16S, df_metagenomic
    except Exception as e:
        logging.error(f"-----> Error in aligning samples in DataFrames: {e}")
        raise


def remove_samples_no_metadata(md_path: str, df_16S: pd.DataFrame, df_metagenomic: pd.DataFrame):
    """
    Identify and remove sample IDs that are not present in the metadata.

    Parameters:
    md_path (str): The path file to the metadata table.
    df_16S (pd.DataFrame): Input Pandas DataFrame of the 16S table.
    df_metagenomic (pd.DataFrame): Input Pandas DataFrame of the metagenomic table.

    Returns:
    tuple: A tuple containing two processed DataFrames.
    """

    logging.info("STEP 3: Starting to align sample IDs from the 16S and metagenomic tables to metadata table.")

    try: 
        # Read metadata file
        md = pd.read_csv(md_path, sep = '\t', low_memory=False)

        # Find common samples in both tables and in the "SampleID" column of md
        common_samples_16S = df_16S.index[df_16S.index.isin(md['SampleID'])]
        common_samples_metagenomic = df_metagenomic.index[df_metagenomic.index.isin(md['SampleID'])]
        
        # Subset table to include only common samples in metadata
        df_16S = df_16S.loc[common_samples_16S]
        df_metagenomic = df_metagenomic.loc[common_samples_16S]
        
        # Reset md index to SampleID
        md = md.set_index('SampleID').rename_axis(None)

        # Check if done correctly
        if df_16S.index.isin(md.index).all() and df_metagenomic.index.isin(md.index).all():
            # Log shape of DataFrames
            logging.info("16S table shape: " + str(df_16S.shape))
            logging.info("Metagenomic table shape: " + str(df_metagenomic.shape))
            logging.info("-----> COMPLETED: DataFrame samples successfully aligned with metadata.")
        else:
            logging.info("16S table shape: " + str(df_16S.shape))
            logging.info("Metagenomic table shape: " + str(df_metagenomic.shape))
            logging.error("-----> ERROR: DataFrames still have different samples from metadata after processing.")
        
        # Return subsetted tables
        return df_16S, df_metagenomic    
    except Exception as e:
            logging.error(f"-----> Error in aligning samples in DataFrames: {e}")
            raise


def remove_features_below_common_min(df_16S: pd.DataFrame, df_metagenomic: pd.DataFrame):
    """
    Remove any feature, either 16S or WGS, below the per-sample minimum 
    (that is, max(min(16S), min(WGS))), forming a common minimal basis for taxonomy comparison

    Parameters:
    df_16S (pd.DataFrame): Input Pandas DataFrame of the 16S table.
    df_metagenomic (pd.DataFrame): Input Pandas DataFrame of the metagenomic table.

    Returns:
    tuple: A tuple containing two processed DataFrames.
    """

    logging.info("STEP 4: Removing any feature, either 16S or WGS, below the per-sample minimum.")

    try:
        logging.info("-----> Finding max(min(16S), min(WGS))).")

        # Transpose DataFrames
        df_16S = df_16S.transpose()
        df_metagenomic = df_metagenomic.transpose()

        # Calculate the sum for each row (axis=1) for both dataframes
        row_sums_df_16S = df_16S.sum(axis=1)
        row_sums_df_metagenomic = df_metagenomic.sum(axis=1)

        # Find the minimum sum value for each dataframe
        min_sum_df_16S = row_sums_df_16S.min()
        min_sum_df_metagenomic = row_sums_df_metagenomic.min()
        
        # Compare the minimum sum values and return the maximum
        max_of_mins = max(min_sum_df_16S, min_sum_df_metagenomic)

        logging.info("-----> Filtering rows.")
        # Filter rows where the sum is greater than or equal to max(min(16S), min(WGS)))
        df_16S = df_16S[row_sums_df_16S >= max_of_mins]
        df_metagenomic = df_metagenomic[row_sums_df_metagenomic >= max_of_mins]

        # Transpose DataFrames back
        # df_16S = df_16S.transpose()
        # df_metagenomic = df_metagenomic.transpose()

        # Log shape of DataFrames
        logging.info("16S table shape: " + str(df_16S.shape))
        logging.info("Metagenomic table shape: " + str(df_metagenomic.shape))

        logging.info("-----> Saving as BIOM tables.")
        # Convert and save dataframes as BIOM tables
        dataframes = {'df_16S': df_16S, 'df_metagenomic': df_metagenomic}
        for name, table in dataframes.items():
            obs_ids = table.index
            samp_ids = table.columns

            biom_table = biom.table.Table(table.values, observation_ids=obs_ids, sample_ids=samp_ids)
            biom_output_file = f"../tables/{name}_subset.biom"

            with biom_open(biom_output_file, 'w') as f:
                biom_table.to_hdf5(f, generated_by="subsetted tables")
        
        logging.info("-----> COMPLETED: Subsetted DataFrames saved and outputted as BIOM.")
       
       # Return subsetted tables
        return df_16S, df_metagenomic    
    except Exception as e:
            logging.error(f"-----> Error in removing any feature, either 16S or WGS, below the per-sample minimum: {e}")
            raise
    

if __name__ == '__main__':
    try:
        # File paths for the BIOM files
        biom_16S_path = '../tables/16S_feature-table.biom'
        biom_metagenomic_path = '../tables/wol2_filtered_table_zebra.biom'

        # Reading and converting BIOM files
        df_16S, df_metagenomic = read_and_convert_biom(biom_16S_path, biom_metagenomic_path)

        # Align the sample IDs between the two dataframes
        df_16S_aligned, df_metagenomic_aligned = remove_set_differences(df_16S, df_metagenomic)

        # Align the sample IDs between the two dataframes with metadata
        md_path = '../metadata/consolidated_metadata_subset.tsv'
        df_16S_aligned_md, df_metagenomic_aligned_md = remove_samples_no_metadata(md_path, df_16S_aligned, df_metagenomic_aligned)

        # Remove any feature, either 16S or WGS, below the per-sample minimum
        df_16S_feature_basis, df_metagenomic_feature_basis = remove_features_below_common_min(df_16S_aligned_md, df_metagenomic_aligned_md)

    except Exception as e:
        logging.error(f"An error occurred in the main execution: {e}")