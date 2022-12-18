import sys
from de import DE
from shade import shade
from shadeils import shadeils
from function import sphere

DE(
  sphere,
  {'lower': -5, 'upper': 5, 'threshold': 0, 'best': 0},
  2,
  0,
  name_output="test_1",
  run=5,
  replace=True,
  debug=True,
  F=0.6,
  CR=0.9,
  popsize=100
)

shade(
  sphere,
  {'lower': -5, 'upper': 5, 'threshold': 0, 'best': 0},
  2,
  2000,
  name_output="shade_test",
  popsize=100
)

shadeils(
  sphere,
  {'lower': -5, 'upper': 5, 'threshold': 0, 'best': 0},
  100,
  [2],
  sys.stdout,
  popsize=100,
  info_de=100,
  debug=True
)