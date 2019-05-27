import pandas as pd

results_file_paths = [
    'results_v2/result_dataset_organism_resistance_manualRemove.csv',
    'results_v2/result_dataset_organism_resistance_manualRemove_IG.csv',
    'results_v2/result_dataset_organism_resistance_manualRemove_noUseless.csv',
    'results_v2/result_dataset_organism_resistance_manualRemove_noUseless_wrapper.csv',
    'results_v2/result_dataset_organism_resistance_noUseless.csv',
    'results_v2/result_dataset_organism_resistance_noUseless_wrapper.csv',
    'results_v2/result_dataset_organism_resistance.csv',
    'results_v2/result_dataset_organism_resistance_IG.csv',
    'results_v2/result_dataset_organism_resistance_manual.csv'
]

mean_results = pd.DataFrame([])
classifiers_files = pd.DataFrame([])
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
        # Get classifiers that score near to the mean
        classifier_file = dict()
        classifier_file['classifier'] = classifer
        classifier_file['fname'] = df_classifier.iloc[0]['fname']
        min_value = None
        min_index = None
        for index, row in df_classifier.iterrows():
            abs_value = abs(row['kappa'] - means['kappa'])
            if min_value is None or min_value > abs_value:
                min_value = abs_value
                min_index = row
        classifier_file['classifier_fname'] = '{}_{}_fold{}.pkl'.format(min_index['fname'].split('.')[0],
                                                                        min_index['classifier'], min_index['fold'])
        classifier_file['kappa'] = means['kappa']
        classifier_file['fold'] = min_index['fold']
        classifier_file = pd.DataFrame(classifier_file, index=[0])
        classifiers_files = classifiers_files.append(classifier_file, ignore_index=True)

mean_results = mean_results.set_index('fname').drop(columns=['fold'])
mean_results.to_csv('results_v2/media_resultados.csv')

classifiers_files = classifiers_files.set_index('fname')
classifiers_files.to_csv('results_v2/classifiers.csv')