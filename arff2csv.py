import arff
import pandas as pd
import csv

arff_file_paths = [
    # 'arffs/dataset_organism_resistance_manualRemove.arff',
    # 'arffs/dataset_organism_resistance_manualRemove_IG.arff',
    # 'arffs/dataset_organism_resistance_manualRemove_noUseless.arff',
    # 'arffs/dataset_organism_resistance_manualRemove_noUseless_wrapper.arff',
    # 'arffs/dataset_organism_resistance_noUseless.arff',
    # 'arffs/dataset_organism_resistance_noUseless_wrapper.arff',
    # 'arffs/dataset_organism_resistance.arff',
    # 'arffs/dataset_organism_resistance_IG.arff',
    'arffs/dataset_organism_resistance_noUseless_top200.arff',
    'arffs/dataset_organism_resistance_manualRemove_noUseless_top200.arff'
]

for arff_file in arff_file_paths:
    print(arff_file)
    arff_data = arff.load(open(arff_file))
    columns = [att[0] for att in arff_data['attributes']]
    df = pd.DataFrame(arff_data['data'], columns=columns)
    # df = pd.get_dummies(df)
    # df = df.fillna(0)
    df = df.sort_index(axis=1)
    df.to_csv('arffs/{}.csv'.format(arff_file.split('/')[1].split('.')[0]), quoting=csv.QUOTE_NONNUMERIC)