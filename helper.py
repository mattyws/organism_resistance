import csv
import numpy as np
import pandas as pd
from scipy.stats import randint as sp_randint
from scipy.stats import expon as sp_expon
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree.tree import DecisionTreeClassifier


def get_itemid_from_key(key):
    return key.split("_")[1]

def load_csv_file(csv_file_path, class_label):
    data = pd.read_csv(csv_file_path)
    classes = data[class_label]
    data = data.drop(class_label, axis=1)
    return data, classes

def generate_random_numbers_tuple():
    while True:
        result = []
        size = sp_randint(1, 10).rvs()
        for i in range(size):
            result.append(sp_expon(scale=50).rvs())
        yield tuple(result)

class RandIntMatrix(object):
    def __init__(self, low, high):
        self.low = low
        self.high = high
        self.shape = sp_randint(1, 3).rvs()

    def rvs(self, random_state=None):
        return sp_randint(self.low, self.high).rvs(size=self.shape)

# generate_random_numbers_tuple()

PARAM_DISTS = {
    type(MLPClassifier()).__name__: {'verbose':1, 'max_iter':200, 'learning_rate_init':0.00001},
    type(GaussianNB()).__name__ : {},
    type(LinearSVC()).__name__ : {},
    type(DecisionTreeClassifier()).__name__ : {},
    type(RandomForestClassifier()).__name__ : {},
    type(LogisticRegression()).__name__ : {}
}

YESNO_LABEL = "yes/no"
CATEGORICAL_LABEL = "categorical"
NUMERIC_LABEL = "numeric"

FEATURES_ITEMS_LABELS = {
    # '5656' : 'Phenylephrine',
    # '221749' : 'Phenylephrine',
    # '6752' : 'Phenylephrine',
    # '221906' : 'Norepinephrine',
    # '1136' : 'Vasopressin',
    # '1222' : 'Vasopressin',
    # '2445' : 'Vasopressin',
    # '30051' : 'Vasopressin',
    # '222315' : 'Vasopressin',
    '220052' : 'Arterial Blood Pressure mean',
    '52' : 'Arterial Blood Pressure mean',
    '3312' : 'Arterial Blood Pressure mean',
    '5679' : 'Arterial Blood Pressure mean',
    '225312' : 'Arterial Blood Pressure mean',
    '5600' : 'Arterial Blood Pressure mean',
    '3256' : 'Arterial Blood Pressure mean',
    '3314' : 'Arterial Blood Pressure mean',
    '3316' : 'Arterial Blood Pressure mean',
    '3320' : 'Arterial Blood Pressure mean',
    '3322' : 'Arterial Blood Pressure mean',
    '5731' : 'Arterial Blood Pressure mean',
    '2732' : 'Arterial Blood Pressure mean',
    '7618' : 'Arterial Blood Pressure mean',
    '7620' : 'Arterial Blood Pressure mean',
    '7622' : 'Arterial Blood Pressure mean',
    '53' : 'Arterial Blood Pressure mean',
    '443' : 'Arterial Blood Pressure mean',
    '456' : 'NBP MEAN',
    '6702' : 'Arterial Blood Pressure mean',
    '224167' : 'BPs L',
    '224643' : 'BPd L',
    '227243' : 'BPs R',
    '227242' : 'BPd R',
    '228332' : 'Delirium assessment',
    '220739' : 'GCS - Eye Opening',
    '184' : 'GCS - Eye Opening',
    '223901' : 'GCS - Motor Response',
    '454' : 'GCS - Motor Response',
    '223900' : 'GCS - Verbal Response',
    '723' : 'GCS - Verbal Response',
    '211' : 'Heart Rate',
    '220045' : 'Heart Rate',
    '223835' : 'Inspired O2 Fraction',
    '448' : 'Minute Volume',
    '224687' : 'Minute Volume',
    '220181' : 'Non Invasive Blood Pressure mean',
    '226732' : 'O2 Delivery Device(s)',
    '467' : 'O2 Delivery Device(s)',
    '220277' : 'O2 saturation pulseoxymetry',
    '1046' : 'Pain Present',
    '223781' : 'Pain Present',
    '535' : 'Peak Insp. Pressure',
    '224695' : 'Peak Insp. Pressure',
    '505' : 'PEEP',
    '6924' : 'PEEP',
    '543' : 'Plateau Pressure',
    '224696' : 'Plateau Pressure',
    '616' : 'Respiratory Effort',
    '223990' : 'Respiratory Effort',
    '618' : 'Respiratory Rate',
    '220210' : 'Respiratory Rate',
    '224690' : 'Respiratory Rate (Total)',
    '615' : 'Respiratory Rate (Total)',
    '676' : 'Temperature Celsius',
    '223762' : 'Temperature Celsius',
    '678' : 'Temperature Celsius',
    '223761' : 'Temperature Celsius',
    '227687' : 'Tobacco Use History',
    '225108' : 'Tobacco Use History',
    '720' : 'Ventilator Mode',
    '223849' : 'Ventilator Mode',
    '646' : 'SPO2',
    '807' : 'Fingerstick glucose',
    '811' : 'Fingerstick glucose',
    '1529' : 'Fingerstick glucose',
    '3745' : 'Fingerstick glucose',
    '3744' : 'Fingerstick glucose',
    '225664' : 'Fingerstick glucose',
    '220621' : 'Fingerstick glucose',
    '226537' : 'Fingerstick glucose'
}

FEATURES_ITEMS_TYPE = {
    # '5656' : YESNO_LABEL,
    # '221749' : YESNO_LABEL,
    # '6752' : YESNO_LABEL,
    # '221906' : YESNO_LABEL,
    # '1136' : YESNO_LABEL,
    # '1222' : YESNO_LABEL,
    # '2445' : YESNO_LABEL,
    # '30051' : YESNO_LABEL,
    # '222315' : YESNO_LABEL,
    '220052' : NUMERIC_LABEL,
    '52' : NUMERIC_LABEL,
    '3312' : NUMERIC_LABEL,
    '5679' : NUMERIC_LABEL,
    '225312' : NUMERIC_LABEL,
    '5600' : NUMERIC_LABEL,
    '3256' : NUMERIC_LABEL,
    '3314' : NUMERIC_LABEL,
    '3316' : NUMERIC_LABEL,
    '3320' : NUMERIC_LABEL,
    '3322' : NUMERIC_LABEL,
    '5731' : NUMERIC_LABEL,
    '2732' : NUMERIC_LABEL,
    '7618' : NUMERIC_LABEL,
    '7620' : NUMERIC_LABEL,
    '7622' : NUMERIC_LABEL,
    '53' : NUMERIC_LABEL,
    '443' : NUMERIC_LABEL,
    '456' : NUMERIC_LABEL,
    '6702' : NUMERIC_LABEL,
    '224167' : NUMERIC_LABEL,
    '224643' : NUMERIC_LABEL,
    '227243' : NUMERIC_LABEL,
    '227242' : NUMERIC_LABEL,
    '228332' : CATEGORICAL_LABEL, # YES_NO
    '220739' : CATEGORICAL_LABEL, # Era MEAN
    '184' : CATEGORICAL_LABEL, # --
    '223901' : CATEGORICAL_LABEL, # --
    '454' : CATEGORICAL_LABEL, # --
    '223900' : CATEGORICAL_LABEL, # --
    '723' : CATEGORICAL_LABEL, # --
    '211' : NUMERIC_LABEL,
    '220045' : NUMERIC_LABEL,
    '223835' : NUMERIC_LABEL,
    '448' : NUMERIC_LABEL,
    '224687' : NUMERIC_LABEL,
    '220181' : NUMERIC_LABEL,
    '226732' : CATEGORICAL_LABEL,
    '467' : CATEGORICAL_LABEL,
    '615' : NUMERIC_LABEL,
    '220277' : NUMERIC_LABEL,
    '1046' : CATEGORICAL_LABEL, # YESNO
    '223781' : CATEGORICAL_LABEL, # YESNO
    '535' : NUMERIC_LABEL,
    '224695' : NUMERIC_LABEL,
    '505' : NUMERIC_LABEL,
    '6924' : NUMERIC_LABEL,
    '543' : NUMERIC_LABEL,
    '224696' : NUMERIC_LABEL,
    '616' : CATEGORICAL_LABEL, # YESNO
    '223990' : CATEGORICAL_LABEL, # YESNO
    '618' : NUMERIC_LABEL,
    '220210' : NUMERIC_LABEL,
    '224690' : NUMERIC_LABEL,
    '676' : NUMERIC_LABEL,
    '223762' : NUMERIC_LABEL,
    '678' : NUMERIC_LABEL,
    '223761' : NUMERIC_LABEL,
    '227687' : CATEGORICAL_LABEL, # YESNO
    '225108' : CATEGORICAL_LABEL, # YESNO
    '720' : CATEGORICAL_LABEL,
    '223849' : CATEGORICAL_LABEL,
    '646' : NUMERIC_LABEL,
    '807': NUMERIC_LABEL,
    '811': NUMERIC_LABEL,
    '1529': NUMERIC_LABEL,
    '3745': NUMERIC_LABEL,
    '3744': NUMERIC_LABEL,
    '225664': NUMERIC_LABEL,
    '220621': NUMERIC_LABEL,
    '226537': NUMERIC_LABEL
}

ARE_EQUAL_CHART = [
    ('723', '223900'), # -- GCSVerbal/GCS - Verbal Response
    ('454', '223901'), # -- GCSMotor/GCS - Motor Response
    ('184', '220739'), #  -- GCSEyes/GCS - Eye Opening
    ('211', '220045'), # -- HEART RATE
    ('220052', '220181'), # -- MEAN ARTERIAL PRESSURE
    ('220052', '52'),
    ('220052', '3312'),
    ('220052', '5679'),
    ('220052', '225312'),
    ('220052', '5600'),
    ('220052', '3256'),
    ('220052', '3314'),
    ('220052', '3316'),
    ('220052', '3320'),
    ('220052', '3322'),
    ('220052', '5731'),
    ('220052', '2732'),
    ('220052', '7618'),
    ('220052', '7620'),
    ('220052', '7622'),
    ('220052', '53'),
    ('220052', '443'),
    ('220052', '456'),
    ('220052', '6702'),

    ('618', '224690'), # -- RESPIRATORY RATE
    ('618', '615'),
    ('618', '220210'),

    ('646', '220277'), # -- SPO2, peripheral

    ('807', '811'), # -- GLUCOSE, both lab and fingerstick
    ('807', '1529'),
    ('807', '3745'),
    ('807', '3744'),
    ('807', '225664'),
    ('807', '220621'),
    ('807', '226537'),

    ('467', '226732'), # -- O2 Delivery Device
    ('467', '468'),
    ('448', '224687'), # -- Minute Volume

    ('1046', '223781'), # -- Pain Present
    ('535', '224695'), # -- Peak Insp. Pressure
    ('505', '6924' ), # -- PEEP
    ('543', '224696'), # -- Plateau Pressure
    ('616', '223990'), # -- Respiratory Effort

    ('720', '223849'), # -- Ventilator Mode
    ('676', '678'), # -- Temperature Celsius
    ('676', '223761'),
    ('676', '223762'),

    ('225108', '227687'), # -- Tobacco Use History
]

FEATURES_LABITEMS_LABELS = {
    '50861' : 'Alanine Aminotransferase (ALT)',
    '50862' : 'Albumin',
    '50863' : 'Alkaline Phosphatase',
    '50801' : 'Alveolar-arterial Gradient',
    '50866' : 'Ammonia',
    '50868' : 'Anion Gap',
    '50878' : 'Asparate Aminotransferase (AST)',
    '51144' : 'Bands',
    '50802' : 'Base Excess',
    '50882' : 'Bicarbonate',
    '50803' : 'Bicarbonate',
    '50885' : 'Bilirubin, Total',
    '50893' : 'Calcium, Total',
    '50902' : 'Chloride',
    '50806' : 'Chloride',
    '50910' : 'Creatine Kinase (CK)',
    '50912' : 'Creatinine',
    '51200' : 'Eosinophils',
    '50809' : 'Glucose',
    '51221' : 'Hematocrit',
    '51222' : 'Hemoglobin',
    '50811' : 'Hemoglobin',
    '51237' : 'INR(PT)',
    '50813' : 'Lactate',
    '50954' : 'Lactate Dehydrogenase (LD)',
    '51244' : 'Lymphocytes',
    '50960' : 'Magnesium',
    '51256' : 'Neutrophils',
    '50963' : 'NTproBNP',
    '50816' : 'Oxygen',
    '50817' : 'Oxygen Saturation',
    '50818' : 'pCO2',
    '50820' : 'pH',
    '50970' : 'Phosphate',
    '51265' : 'Platelet Count',
    '50821' : 'pO2',
    '50971' : 'Potassium',
    '50822' : 'Potassium, Whole Blood',
    '50889' : 'Protein',
    '51274' : 'PT',
    '51275' : 'PTT',
    '51277' : 'RDW',
    '51279' : 'Red Blood Cells',
    '50983' : 'Sodium',
    '51003' : 'Troponin T',
    '51006' : 'Urea Nitrogen',
    '51301' : 'White Blood Cells',
    '50824' : 'Sodium, Whole Blood',
    '50931' : 'Glucose',
    '50810' : 'Hematocrit',
    '51300' : 'WBC Count'
}

FEATURES_LABITEMS_TYPE = {
    '50861' : NUMERIC_LABEL,
    '50862' : NUMERIC_LABEL,
    '50863' : NUMERIC_LABEL,
    '50801' : NUMERIC_LABEL,
    '50866' : NUMERIC_LABEL,
    '50868' : NUMERIC_LABEL,
    '50878' : NUMERIC_LABEL,
    '51144' : NUMERIC_LABEL,
    '50802' : NUMERIC_LABEL,
    '50882' : NUMERIC_LABEL,
    '50803' : NUMERIC_LABEL,
    '50885' : NUMERIC_LABEL,
    '50893' : NUMERIC_LABEL,
    '50902' : NUMERIC_LABEL,
    '50806' : NUMERIC_LABEL,
    '50910' : NUMERIC_LABEL,
    '50912' : NUMERIC_LABEL,
    '51200' : NUMERIC_LABEL,
    '50809' : NUMERIC_LABEL,
    '51221' : NUMERIC_LABEL,
    '51222' : NUMERIC_LABEL,
    '50811' : NUMERIC_LABEL,
    '51237' : NUMERIC_LABEL,
    '50813' : NUMERIC_LABEL,
    '50954' : NUMERIC_LABEL,
    '51244' : NUMERIC_LABEL,
    '50960' : NUMERIC_LABEL,
    '51256' : NUMERIC_LABEL,
    '50963' : NUMERIC_LABEL,
    '50816' : NUMERIC_LABEL,
    '50817' : NUMERIC_LABEL,
    '50818' : NUMERIC_LABEL,
    '50820' : NUMERIC_LABEL,
    '50970' : NUMERIC_LABEL,
    '51265' : NUMERIC_LABEL,
    '50821' : NUMERIC_LABEL,
    '50971' : NUMERIC_LABEL,
    '50822' : NUMERIC_LABEL,
    '50889' : NUMERIC_LABEL,
    '51274' : NUMERIC_LABEL,
    '51275' : NUMERIC_LABEL,
    '51277' : NUMERIC_LABEL,
    '51279' : NUMERIC_LABEL,
    '50983' : NUMERIC_LABEL,
    '51003' : NUMERIC_LABEL,
    '51006' : NUMERIC_LABEL,
    '51301' : NUMERIC_LABEL,
    '50824' : NUMERIC_LABEL,
    '50931' : NUMERIC_LABEL,
    '50810' : NUMERIC_LABEL,
    '51300' : NUMERIC_LABEL
}


ARE_EQUAL_LAB = [
    ('50806', '50902'),
    ('50809', '50931'),
    ('50810', '51221'),
    ('50811', '51222'),
    ('50822', '50971'),
    ('50824', '50983'),
    ('51300', '51301'),
    ('50803', '50882'),
]

FARENHEIT_ID = ['678', '223761']
PRESSURE_IDS = ['220052', '220181', '52', '3312', '5679', '225312', '5600', '3256', '3314', '3316', '3320',
                '3322', '5731', '2732', '7618', '7620', '7622', '53', '443', '456', '6702']

SYSTOLIC_IDS = ['51', '442', '455', '6701', '220179', '220050']
DIASTOLIC_IDS = ['8368', '8440', '8441', '8555', '220180', '220051']

CELCIUS = lambda Tf : ((Tf - 32)*5)/9
MEAN_PRESSURE = lambda D, S: (2*D + S)/3

def get_attributes_from_arff(file_name):
    arff_file = open(file_name, 'r')
    attributes = []
    start_attribute = False
    att_types = dict()
    for line in arff_file:
        line = line.strip()
        if not start_attribute and line.startswith('@attribute'):
            start_attribute = True
        elif start_attribute and line.startswith('@attribute'):
            att_name = line.split(' ')[1]
            attributes.append(att_name)
            att_type = line.split(' ')[2]
            if att_type == 'numeric':
                att_types[att_name] = NUMERIC_LABEL
            else:
                att_types[att_name] = CATEGORICAL_LABEL
        elif start_attribute and not line.startswith('@attribute'):
            break
    return attributes, att_types

def get_attributes_from_csv(file_name):
    df = pd.read_csv(file_name)
    return list(df.columns)
