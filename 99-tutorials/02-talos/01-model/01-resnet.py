"""This tutorial contains 2 ways to construct a resnet using tframe API.
The basic building blocks to be demonstrated have a general form of
        x_{l+1} = x_l + F(x_l),
where F is an arbitrary function whose output has the same shape with x_l.
In this demo F is chosen as a regular Conv layer.
"""
from tframe import mu
from tframe import Predictor

import os


# -----------------------------------------------------------------------------
# Define 2 functions for create res-blocks
# -----------------------------------------------------------------------------
def get_res_block_using_shortcut(filters, kernel_size, activation):
  return mu.ShortCut(mu.Conv2D(filters, kernel_size, activation=activation),
                     mode=mu.ShortCut.Mode.SUM)

def get_res_block_using_DAG(filters, kernel_size, activation):
  return mu.ForkMergeDAG(
    [mu.Conv2D(filters, kernel_size, activation=activation),  # vertex 1
     mu.Merge.Sum()],                                         # vertex 2
    edges='1;11', name='ResBlock')

# -----------------------------------------------------------------------------
# Initialize a tframe.Predictior
# -----------------------------------------------------------------------------
model = Predictor(mark='resnet')
model.add(mu.Input(sample_shape=[32, 32, 3]))
h1 = model.add(mu.Conv2D(8, kernel_size=3))

# -----------------------------------------------------------------------------
# Add residual blocks
# -----------------------------------------------------------------------------
funcs = [get_res_block_using_shortcut, get_res_block_using_DAG]
for _ in range(2):
  model.add(funcs[1](filters=8, kernel_size=3, activation='relu'))

# -----------------------------------------------------------------------------
# Rehearse
# -----------------------------------------------------------------------------
model.rehearse(
  export_graph=True, build_model=True, path=os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '01-resnet'))
