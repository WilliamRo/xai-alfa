from pictor.xomics import FeatureExplorer, Omix
from pictor.xomics.ml.logistic_regression import LogisticRegression

import os



# -----------------------------------------------------------------------------
# Load features and targets
# -----------------------------------------------------------------------------
data_dir = r'../../data/'
file_names = ['prostate-305-clinic.omix',
              'prostate-305-radiomics.omix']

cli_omix: Omix = Omix.load(os.path.join(data_dir, file_names[0]))
rad_omix: Omix = Omix.load(os.path.join(data_dir, file_names[1]))

cli_omix.target_labels[0] = 'TRG<=3'
rad_omix.target_labels[0] = 'TRG<=3'
cli_omix.target_labels[1] = 'TRG>3'
rad_omix.target_labels[1] = 'TRG>3'

# -----------------------------------------------------------------------------
# Load features and targets
# -----------------------------------------------------------------------------
omix = rad_omix

# (1) Select features
omix = omix.select_features('lasso', verbose=1)

# omix.show_in_explorer()

# (2) Machine learning and evaluation
lr = LogisticRegression(
  ignore_warnings=0,
)

lr.fit_k_fold(
  omix,
  verbose=1,

  cm=1,
  print_cm=0,

  auc=1,
  plot_roc=0,
  random_state=1219,
)
