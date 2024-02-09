import pandas as pd
import biom
from biom import load_table
from biom.util import biom_open
import numpy as np
from numpy.random import RandomState
from numpy import array
import qiime2 as q2
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import logging

### This script performs the rarefaction to adjust for differences in library sizes across samples from the subsetted tables

# Setup logging
logging.basicConfig(filename='../logs/rarefy_tables.log', level=logging.DEBUG,
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


def rarefy_table(biom_df: pd.DataFrame, sampling_depth: int, seed: int = 42) -> pd.DataFrame:
    """
    Helper function to rarefy a single biological observation matrix (biom) table based on a specified sampling depth.

    Parameters:
    - biom_df (pd.DataFrame): Input Pandas DataFrame representing the biom table. Rows should represent samples, 
                              and columns should represent observations (e.g., species, genes).
    - sampling_depth (int): The target sampling depth for rarefaction.
    - seed (int): Seed for the random number generator to ensure reproducibility. Default is 42.

    Returns:
    pd.DataFrame: A rarefied DataFrame where each sample has been randomly subsampled to the specified sampling depth
                  across all samples in the input DataFrame.
    """

    # Convert DataFrame to numpy array for rarefaction
    data_arr = biom_df.values
    
    # Initialize random number generator
    prng = np.random.RandomState(seed)
    
    # Rarefaction process
    rarefied_data = np.zeros_like(data_arr) # Initialize of the rarefied data array
    for i, row in enumerate(data_arr):
        total_count = row.sum()
        if total_count >= sampling_depth:  # Ensure the row has enough counts for rarefaction
            probabilities = row / total_count
            chosen_indices = prng.choice(biom_df.columns.size, sampling_depth, p=probabilities)
            rarefied_row = np.bincount(chosen_indices, minlength=biom_df.columns.size)
            rarefied_data[i] = rarefied_row
    
    return pd.DataFrame(rarefied_data, index=biom_df.index, columns=biom_df.columns)


def rarefy_tables(df_16S: pd.DataFrame, df_metagenomic: pd.DataFrame) -> tuple:
    """
    Perform rarefaction on both 16S rRNA gene sequencing and metagenomic sequencing data tables,
    adjusting each table to a specific sampling depth based on their respective needs.

    Parameters:
    - df_16S (pd.DataFrame): Input Pandas DataFrame of the 16S rRNA gene sequencing table.
    - df_metagenomic (pd.DataFrame): Input Pandas DataFrame of the metagenomic sequencing table.

    Returns:
    tuple: A tuple containing two rarefied dataframes:
           - The first DataFrame corresponds to the rarefied 16S rRNA biom table.
           - The second DataFrame corresponds to the rarefied metagenomic biom table.
    """

    logging.info("STEP 2: Rarefying BIOM tables to specified sampling depths.")

    # Find min sampling depth
    # sampling_depth_tbl_16S = pd.DataFrame(index = df_16S.index)
    # sampling_depth_tbl_16S['sampling_depth'] = df_16S.sum(axis = 1)
    # sampling_depth_sorted_tbl_16S = sampling_depth_tbl_16S.sort_values(by='sampling_depth', ascending = False)
    # min_depth_16S = int(min(sampling_depth_sorted_tbl_16S['sampling_depth']))
    # logging.info(f"Minimum sampling depth for 16S {min_depth_16S}")

    # sampling_depth_tbl_metagenomic = pd.DataFrame(index = df_metagenomic.index)
    # sampling_depth_tbl_metagenomic['sampling_depth'] = df_metagenomic.sum(axis = 1)
    # sampling_depth_sorted_tbl_metagenomic = sampling_depth_tbl_metagenomic.sort_values(by='sampling_depth', ascending = False)
    # min_depth_metagenomic = int(min(sampling_depth_sorted_tbl_metagenomic['sampling_depth']))
    # logging.info(f"Minimum sampling depth for metagenomic {min_depth_metagenomic}")

    # Rarefy each table with specified sampling depths
    rarefied_16S = rarefy_table(df_16S, sampling_depth=9000)
    rarefied_metagenomic = rarefy_table(df_metagenomic, sampling_depth=2000000)

    # rarefied_16S = rarefy_table(df_16S, sampling_depth=100)
    # rarefied_metagenomic = rarefy_table(df_metagenomic, sampling_depth=100)
    
    # rarefied_16S = rarefy_table(df_16S, sampling_depth=min_depth_16S)
    # rarefied_metagenomic = rarefy_table(df_metagenomic, sampling_depth=min_depth_metagenomic)

    # Transpose back to features by samples
    rarefied_16S = rarefied_16S.transpose()
    rarefied_metagenomic = rarefied_metagenomic.transpose()

    # Convert and save dataframes as BIOM tables
    dataframes = {'df_16S': rarefied_16S, 'df_metagenomic': rarefied_metagenomic}
    for name, table in dataframes.items():
        obs_ids = table.index
        samp_ids = table.columns

        biom_table = biom.table.Table(table.values, observation_ids=obs_ids, sample_ids=samp_ids)
        biom_output_file = f"../tables/{name}_subset_rare_9k2M.biom"

        with biom_open(biom_output_file, 'w') as f:
            biom_table.to_hdf5(f, generated_by="subsetted tables")
    
    logging.info("-----> COMPLETED: Rarefied DataFrames saved and outputted as BIOM.")
    return (rarefied_16S, rarefied_metagenomic)



if __name__ == '__main__':
    logging.basicConfig(filename='../logs/rarefy_tables.log', level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    try:
        # Define file paths to your BIOM tables
        biom_16S_path = '../tables/df_16S_subset.biom'
        biom_metagenomic_path = '../tables/df_metagenomic_subset.biom'
        
        # Read and convert BIOM tables to Pandas DataFrames
        df_16S, df_metagenomic = read_and_convert_biom(biom_16S_path, biom_metagenomic_path)
        
        # Perform rarefaction and get rarefied tables
        rarefied_16S, rarefied_metagenomic = rarefy_tables(df_16S, df_metagenomic)
        
        logging.info("Rarefaction completed successfully.")
        
        # If you have additional steps to save or process the rarefied tables, include them here
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise
