import pandas as pd
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import scipy
import scipy.stats as ss
from skbio.stats.distance import permanova
import biom
from biom import load_table
from biom.util import biom_open
from gemelli.rpca import auto_rpca
import logging

### This script merges the 16S and metagenomic tables collapsed by taxonomy levels

# Setup logging
logging.basicConfig(filename='../logs/merge_16S_metagenomic_tables.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')



def merge_16S_metagenomic_tables(taxon_level: str):
    """
    Merges 16S and metagenomic tables for a specified taxonomic level.

    Parameters:
    - taxon_level: The taxonomic level to merge ('Family', 'Genus', 'Species').

    Returns:
    - A merged DataFrame of 16S and metagenomic data for the specified taxonomic level.
    """

    logging.info(f"-----> Merging {taxon_level} level 16S and metagenomic tables")

    taxon_list = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    
    if taxon_level not in taxon_list:
        raise ValueError(f"taxon_level must be one of {taxon_list}")
    
    df_16S_path = f"../tables/df_16S_collapsed_{taxon_level}_thdmi.biom"
    df_metagenomic_path = f"../tables/df_metagenomic_collapsed_{taxon_level}_thdmi.biom"

    try:
        biom_table_16S = load_table(df_16S_path)
        df_16S = pd.DataFrame(biom_table_16S.to_dataframe())
        df_16S = df_16S.add_suffix('_16S')
        logging.info(f"16S table shape is: {df_16S.shape}")

        biom_table_metagenomic = load_table(df_metagenomic_path)
        df_metagenomic = pd.DataFrame(biom_table_metagenomic.to_dataframe())
        df_metagenomic = df_metagenomic.add_suffix('_MG')
        logging.info(f"Metagenomic table shape is: {df_metagenomic.shape}")

        # Example merge operation, adjust according to your data structure
        df_merged = df_16S.merge(df_metagenomic, left_index=True, right_index=True)
        logging.info(f"Merged table shape is: {df_merged.shape}")

        # Convert and save dataframes as BIOM tables
        obs_ids = df_merged.index
        samp_ids = df_merged.columns

        biom_table = biom.table.Table(df_merged.values, observation_ids=obs_ids, sample_ids=samp_ids)
        biom_output_file = f"../tables/df_merged_16S_metagenomic_{taxon_level}.biom"

        with biom_open(biom_output_file, 'w') as f:
            biom_table.to_hdf5(f, generated_by="merged 16S and metagenomic tables of taxonomy collapsed tables")

        logging.info(f"Table merged and saved as BIOM")

        return df_merged
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return None       
    

if __name__ == '__main__':
    try:
        merge_16S_metagenomic_tables('Kingdom')
        merge_16S_metagenomic_tables('Phylum')
        merge_16S_metagenomic_tables('Class')
        merge_16S_metagenomic_tables('Order')
        merge_16S_metagenomic_tables('Family')
        merge_16S_metagenomic_tables('Genus')
        merge_16S_metagenomic_tables('Species')
        logging.info(f"-----> COMPLETED: All tables merged") 

    except Exception as e:
        logging.error(f"An error occurred in the main execution: {e}")    