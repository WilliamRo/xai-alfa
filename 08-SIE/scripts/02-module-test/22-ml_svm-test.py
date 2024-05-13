from pictor import Pictor
from pictor.xomics import FeatureExplorer, Omix
from pictor.xomics.ml.support_vector_machine import SupportVectorMachine
from sie.misc import load_features_and_targets

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
omix = cli_omix * rad_omix

svm = SupportVectorMachine(
  ignore_warnings=1,
)

svm.fit_k_fold(
  cli_omix,
  verbose=1,

  cm=1,
  print_cm=0,

  auc=1,
  plot_roc=0,
  random_state=1219,
)
"""
random_state=1219,
:: Evaluation Result (Prostate-305-clinic):
------------------------------------------------------
               Precision    Recall  F1-Score  Support 
======================================================
 TRG<=3           0.8432    0.9567    0.8964      208 
 TRG>3            0.8696    0.6186    0.7229       97 
------------------------------------------------------
 Accuracy                             0.8492      305 
 Macro Avg        0.8564    0.7876    0.8096      305 
 Weighted Avg     0.8516    0.8492    0.8412      305 
------------------------------------------------------
:: AUC (Prostate-305-clinic) = 0.907 
"""




