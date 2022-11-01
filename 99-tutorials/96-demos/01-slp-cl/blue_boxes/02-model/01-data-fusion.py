from slp_core import th

import slp_mu



# Configure
th.data_config = '7'
th.aug_config = ''
th.channels = '0,1'

th.mark = 'data-fusion'

# Get model and rehearse
model = slp_mu.get_data_fusion_model()
model.rehearse(export_graph=True, build_model=False)



