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

from pictor.xomics.ml.random_forest import RandomForestClassifier

model = RandomForestClassifier(
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
[TUNE] >> Best hyperparameters: {'criterion': 'entropy', 'max_depth': 6, 'max_features': None, 'n_estimators': 100}
[TUNE] >> Best score: 0.8656

:: Evaluation Result (Prostate-305-clinic):
------------------------------------------------------
               Precision    Recall  F1-Score  Support 
======================================================
 TRG<=3           0.8694    0.9279    0.8977      208 
 TRG>3            0.8193    0.7010    0.7556       97 
------------------------------------------------------
 Accuracy                             0.8557      305 
 Macro Avg        0.8443    0.8145    0.8266      305 
 Weighted Avg     0.8534    0.8557    0.8525      305 
------------------------------------------------------
:: AUC (Prostate-305-clinic) = 0.914
"""




