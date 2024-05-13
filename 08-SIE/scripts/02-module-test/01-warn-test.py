from pictor import Pictor
from pictor.xomics import FeatureExplorer, Omix
from sie.misc import load_features_and_targets

import os
import pictor.xomics.ml.logistic_regression as lr



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

p, fe = FeatureExplorer.explore(
  omix=cli_omix, title=cli_omix.data_name, auto_show=False,
  ignore_warnings=0)
fe.sf_lasso(verbose=0)



