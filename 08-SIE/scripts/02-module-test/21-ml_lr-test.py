from pictor import Pictor
from pictor.xomics import FeatureExplorer, Omix
from pictor.xomics.ml.logistic_regression import LogisticRegression
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

# cli_omix.show_in_explorer()
# rad_omix.show_in_explorer()

# hp = lr.tune(omix, verbose=1, n_jobs=10)
# hp = lr.fit(cli_omix, hp_verbose=1, verbose=0, cm=True)
lr = LogisticRegression(
  ignore_warnings=1,
)

pkg = lr.fit_k_fold(
  cli_omix,
  verbose=1,

  cm=1,
  print_cm=1,
  plot_cm=0,
  mi=0,

  auc=1,
  plot_roc=0,
  random_state=1219,
)

pkg_post = pkg.evaluate(cli_omix)
pkg_post.report()

"""
seed = 1219
------------------------------------------------------
               Precision    Recall  F1-Score  Support 
======================================================
 TRG<=3           0.8395    0.9808    0.9047      208 
 TRG>3            0.9355    0.5979    0.7296       97 
------------------------------------------------------
 Accuracy                             0.8590      305 
 Macro Avg        0.8875    0.7894    0.8171      305 
 Weighted Avg     0.8700    0.8590    0.8490      305 
------------------------------------------------------
:: AUC (Prostate-305-clinic) = 0.913
"""

