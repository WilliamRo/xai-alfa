from pictor import Pictor
from pictor.xomics import FeatureExplorer, Omix
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

cli_omix.show_in_explorer()
# rad_omix.show_in_explorer()

# omix.show_in_explorer()
# cli_omix.show_in_explorer()
# rad_omix.show_in_explorer()

# omices = omix.split(1, 1, 1)
# for o in omices: o.report()
# print()
# print(sum(omices[1:], start=omices[0]).report())









