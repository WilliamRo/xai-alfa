from md_core import th
from tframe import mu



def get_container(time_steps=1000) -> mu.Predictor:
  model = mu.GaussianDiffusion(
    mark=th.mark, x_shape=th.input_shape, time_steps=time_steps,
    beta_schedule=th.beta_schedule)
  return model



def finalize(model):
  assert isinstance(model, mu.Predictor)

  model.add(mu.Conv2D(th.input_shape[-1], kernel_size=1))

  # Build model
  model.build(loss='MSE', metric=['loss'])
  return model



class TimeUNet2D(mu.UNet):

  def __init__(self, arc_string, diffusion_model: mu.GaussianDiffusion,
               **kwargs):
    super().__init__(dimension=2, arc_string=arc_string, **kwargs)
    self.model = diffusion_model



  def _get_layers(self):
    """Abstract method defined by ConvNet, returns a list of layers"""
    layers, floors = [], []

    # Define some utilities
    contract = lambda channels: layers.append(
      self._get_pooling() if self.use_maxpool
      else self._get_conv(channels, self.contraction_kernel_size, strides=2))
    expand = lambda channels: layers.append(self._get_conv(
      channels, self.expansion_kernel_size, strides=2, transpose=True))

    # (1) Build left tower for contracting
    filters = self.filters
    for i in range(self.height):  # (height - i)-th floor
      # Add front layers on each floor
      for _ in range(self.thickness):
        layers.append(self._get_conv(filters, self.contraction_kernel_size))
      # Register the last layer in each floor before contracting
      floors.append(layers[-1])

      # TODO: (1)
      layers.append(self.model.get_time_emb_layer())

      # Contract
      contract(filters)
      # Double filters
      filters *= 2

    # (2) Build ground floor (GF)
    for _ in range(self.thickness):
      layers.append(self._get_conv(filters, self.expansion_kernel_size))

    # (3) Build right tower for expanding
    for i in range(1, self.height + 1):  # i-th floor
      # Halve filters
      filters = filters // 2
      # Expand
      expand(filters)
      # Build a bridge if necessary
      if i in self.link_indices:
        guest_is_larger = None
        if self.auto_crop: guest_is_larger = not self.use_maxpool
        layers.append(mu.Bridge(floors[self.height - i], guest_is_larger,
                             guest_first=self.guest_first))
        if self.bottle_net_after_bridge:
          layers.append(self._get_conv(filters, kernel_size=1))
      # Increase thickness
      for _ in range(self.thickness):
        layers.append(self._get_conv(filters, self.expansion_kernel_size))

      # TODO: (2)
      layers.append(self.model.get_time_emb_layer())

    return layers
