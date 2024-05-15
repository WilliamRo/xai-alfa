from pictor.xomics import FeatureExplorer, Omix

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

from pictor.xomics.ml.xgboost import XGBClassifier

model = XGBClassifier(
  ignore_warnings=0,
  n_jobs= 1,
)

model.fit_k_fold(
  cli_omix,
  verbose=1,

  cm=1,
  print_cm=0,

  auc=1,
  plot_roc=0,
  random_state=1219,
)

"""random_state=1219,
Fitting 5 folds for each of 48 candidates, totalling 240 fits
[TUNE] >> Best hyperparameters: {'learning_rate': 0.1, 'max_depth': 4,
          n_estimators': 100, 'subsample': 0.75}
[TUNE] >> Best score: 0.8623
[TUNE] >> Elapsed time: 12.34 seconds

[K_FOLD_FIT] >> Fitting completed.
:: Evaluation Result (Prostate-305-clinic):
------------------------------------------------------
               Precision    Recall  F1-Score  Support 
======================================================
 TRG<=3           0.8796    0.9135    0.8962      208 
 TRG>3            0.7978    0.7320    0.7634       97 
------------------------------------------------------
 Accuracy                             0.8557      305 
 Macro Avg        0.8387    0.8227    0.8298      305 
 Weighted Avg     0.8536    0.8557    0.8540      305 
------------------------------------------------------
:: AUC (Prostate-305-clinic) = 0.916
"""




