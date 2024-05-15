from pictor.xomics import FeatureExplorer, Omix
from pictor.xomics.ml.logistic_regression import LogisticRegression
from pictor.xomics.evaluation.pipeline import Pipeline

import os



# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
data_dir = r'../../data/'
file_name = [
  '20240515-cli-rad.omix',
  '20240515-cli.omix'
][1]

# -----------------------------------------------------------------------------
# Load features and targets
# -----------------------------------------------------------------------------
omix: Omix = Omix.load(os.path.join(data_dir, file_name))

# -----------------------------------------------------------------------------
# Load features and targets
# -----------------------------------------------------------------------------
pi = Pipeline(omix, ignore_warnings=1)
pi.report()


