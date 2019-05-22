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

mean_results = pd.DataFrame([])
for file in results_file_paths:
    print(file)
    df = pd.read_csv(file)
    classifiers = df['classifier'].unique()
    for classifer in classifiers:
        df_classifier = df[df['classifier'] == classifer]
        means = df_classifier.mean()
        means['classifier'] = classifer
        means['fname'] = df_classifier.iloc[0]['fname']
        mean_results = mean_results.append(means, ignore_index=True)
mean_results = mean_results.set_index('fname').drop(columns=['fold', 'auc_b'])
mean_results.to_csv('results/media_resultados.csv')