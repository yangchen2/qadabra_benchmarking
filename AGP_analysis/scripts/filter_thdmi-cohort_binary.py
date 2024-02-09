import pandas as pd
import biom
from biom import load_table
from biom.util import biom_open
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import logging

### This script filters out samples from the collapsed taxonomy tables that do not have either 'US' or 'Mexico' as values under thdmi_cohort
### This was done because previous results from the THDMI dataset show that population has the strongest effect size on the microbiome
### The binary subsetting (US vs Mexico) is done because QADABRA requires binary metadata covariates

# Setup logging
logging.basicConfig(filename='../logs/filter_thdmi-cohort_binary.py.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def filter_thdmi_cohort_binary(biom_paths: list, md_path: str):
    """
    Takes list of BIOM paths for filtering out samples not present in metadata.

    Parameters:
    biom_paths (list): path to BIOM table
    md_path (str): path to metadata file

    Returns:
    nothing returned
    """

    logging.info("STEP 1: Reading in metadata file")
    md = pd.read_csv(md_path, sep='\t')
    
    logging.info("Subsetting to just thdmi_cohort column and either US or Mexico values")
    md_thdmi = md[['SampleID', 'thdmi_cohort']]
    md_thdmi.set_index('SampleID', inplace=True, drop=True)
    md_thdmi.index.name = None
    md_thdmi_cohort_binary = md_thdmi[md_thdmi['thdmi_cohort'].isin(['US', 'Mexico'])]

    logging.info("-----> COMPLETED: Metadata parsed")

    logging.info("STEP 2: Converting BIOMs to dataframes.")
    for biom_path in biom_paths:
        logging.info(f"Table file: {biom_path}")
        biom_table = load_table(biom_path)

        biom_df = pd.DataFrame(biom_table.to_dataframe())
        logging.info(f"Table shape before filter: {biom_df.shape}")
        # Transpose for filtering
        biom_df = biom_df.transpose()

        # Filter the table
        biom_df_filtered = biom_df.loc[biom_df.index.isin(md_thdmi_cohort_binary.index)]
        # Transpose back
        biom_df_filtered = biom_df_filtered.transpose()
        logging.info(f"Table shape after filter: {biom_df_filtered.shape}")

        # Convert backing to BIOM table
        obs_ids = biom_df_filtered.index
        samp_ids = biom_df_filtered.columns
        biom_table = biom.table.Table(biom_df_filtered.values, observation_ids=obs_ids, sample_ids=samp_ids)

        # Split the original path to insert '_thdmi' before the extension
        path_without_extension = biom_path.rsplit('.', 1)[0]
        extension = biom_path.split('.')[-1]

        # Construct the new path with '_thdmi' inserted before the extension
        biom_output_file = f"{path_without_extension}_thdmi.{extension}"
        with biom_open(biom_output_file, 'w') as f:
            biom_table.to_hdf5(f, generated_by="subset to only samples with US or UK under thdmi_cohort")

    logging.info("-----> COMPLETED: All filtered BIOM tables saved.")

if __name__ == '__main__':
    try:
        # File paths for the BIOM files 
        biom_paths = ['../tables/df_16S_collapsed_Kingdom.biom',
                      '../tables/df_16S_collapsed_Phylum.biom',
                      '../tables/df_16S_collapsed_Class.biom',
                      '../tables/df_16S_collapsed_Order.biom',
                      '../tables/df_16S_collapsed_Family.biom',
                      '../tables/df_16S_collapsed_Genus.biom',
                      '../tables/df_16S_collapsed_Species.biom',
                      '../tables/df_metagenomic_collapsed_Kingdom.biom',
                      '../tables/df_metagenomic_collapsed_Phylum.biom',
                      '../tables/df_metagenomic_collapsed_Class.biom',
                      '../tables/df_metagenomic_collapsed_Order.biom',
                      '../tables/df_metagenomic_collapsed_Family.biom',
                      '../tables/df_metagenomic_collapsed_Genus.biom',
                      '../tables/df_metagenomic_collapsed_Species.biom']
        
        md_path = '../metadata/consolidated_metadata_subset.tsv'

        filter_thdmi_cohort_binary(biom_paths, md_path)
    except Exception as e:
        logging.error(f"An error occurred in the main execution: {e}")
