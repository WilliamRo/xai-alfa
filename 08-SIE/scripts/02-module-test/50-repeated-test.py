from pictor.xomics import FeatureExplorer, Omix
from pictor.xomics.ml.logistic_regression import LogisticRegression
from pictor.xomics.evaluation.pipeline import Pipeline

import os



# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
data_dir = r'../../data/'
file_names = ['prostate-305-clinic.omix',
              'prostate-305-radiomics.omix']

# -----------------------------------------------------------------------------
# Load features and targets
# -----------------------------------------------------------------------------
cli_omix: Omix = Omix.load(os.path.join(data_dir, file_names[0]))
rad_omix: Omix = Omix.load(os.path.join(data_dir, file_names[1]))
cli_omix.target_labels[0] = 'TRG<=3'
rad_omix.target_labels[0] = 'TRG<=3'
cli_omix.target_labels[1] = 'TRG>3'
rad_omix.target_labels[1] = 'TRG>3'

# -----------------------------------------------------------------------------
# Load features and targets
# -----------------------------------------------------------------------------
# omix = cli_omix * rad_omix
omix = cli_omix

pi = Pipeline(omix, ignore_warnings=1, save_models=0)

pi.create_sub_space('lasso', repeats=5, show_progress=1)

pi.fit_traverse_spaces('lr', repeats=5, show_progress=1)
pi.fit_traverse_spaces('xgb', repeats=5, show_progress=1)
# pi.fit_traverse_spaces('svm', repeats=5, show_progress=1)

omix.save(os.path.join(data_dir, '20240515-cli.omix'), verbose=True)
