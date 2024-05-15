from pictor import Pictor
from pictor.xomics import FeatureExplorer, Omix
from pictor.xomics.ml.decision_tree import DecisionTree

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

dt = DecisionTree(
  ignore_warnings=0,
  n_jobs= 1,
)

dt.fit_k_fold(
  cli_omix,
  verbose=1,

  cm=1,
  print_cm=0,

  auc=1,
  plot_roc=0,
  random_state=1219,

  # show_signature=1,
)
"""
:: Evaluation Result (Prostate-305-clinic):
:: random_state=1219,

[TUNE] >> Tuning hyperparameters using grid search...
[TUNE] >> ConvergenceWarnings may appear if warning is not ignored.
Fitting 5 folds for each of 24 candidates, totalling 120 fits
[TUNE] >> Best hyperparameters: {'criterion': 'gini', 'max_depth': 4, 'max_features': 'sqrt'}
[TUNE] >> Best score: 0.8525
[TUNE] >> Elapsed time: 0.23 seconds

------------------------------------------------------
               Precision    Recall  F1-Score  Support 
======================================================
 TRG<=3           0.8578    0.8990    0.8779      208 
 TRG>3            0.7586    0.6804    0.7174       97 
------------------------------------------------------
 Accuracy                             0.8295      305 
 Macro Avg        0.8082    0.7897    0.7977      305 
 Weighted Avg     0.8263    0.8295    0.8269      305 
------------------------------------------------------
:: AUC (Prostate-305-clinic) = 0.852
"""




