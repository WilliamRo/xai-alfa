from pictor import Pictor
from pictor.xomics import FeatureExplorer
from sie.misc import load_features_and_targets



# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
data_key = 'gordon-424'

# -----------------------------------------------------------------------------
# Load features and targets
# -----------------------------------------------------------------------------
features, targets = load_features_and_targets(data_key=data_key)

# -----------------------------------------------------------------------------
# Load features and targets
# -----------------------------------------------------------------------------
FeatureExplorer.explore(features, targets, title='Feature Explorer')



