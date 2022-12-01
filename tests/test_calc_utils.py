import tmoutproc
import tmoutproc as top
import pytest
import numpy as np

def test_create_dynamical_matrix():
    #TODO: improve test qualitiy
    test_data_path= "./tests/test_data/"
    dyn_matrix = top.create_dynamical_matrix(f"{test_data_path}/hessian_xtb", f"{test_data_path}/coord_h2", t2SI=False, dimensions=3)
    h_weight = tmoutproc.atom_weight("h")
    assert np.isclose(dyn_matrix[0,0], 0.2680162066/h_weight)
    assert np.isclose(dyn_matrix[0, 1], -0.0000014025 / h_weight)
    assert np.max(np.abs(dyn_matrix-np.transpose(dyn_matrix))) == 0