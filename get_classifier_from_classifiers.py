import pandas as pd

results_file_paths = [
    'results/result_dataset_organism_resistance_manualRemove.csv',
    'results/result_dataset_organism_resistance_manualRemove_IG.csv',
    'results/result_dataset_organism_resistance_manualRemove_noUseless.csv',
    'results/result_dataset_organism_resistance_manualRemove_noUseless_wrapper.csv',
    'results/result_dataset_organism_resistance_noUseless.csv',
    'results/result_dataset_organism_resistance_noUseless_wrapper.csv',
    'results/result_dataset_organism_resistance.csv',
    'results/result_dataset_organism_resistance_IG.csv',
    'results/result_dataset_organism_resistance_manual.csv'
]

results_fold_df = pd.read_csv('results/media_resultados.csv')
