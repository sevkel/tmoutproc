import tmoutproc as top
import pytest
import numpy as np
import os.path

def test_load_xyz_file():
    #problem not acutal code is used but installed code
    coord_xyz = top.read_xyz_file("./tests/test_data/benz.xyz")

    assert coord_xyz.shape == (4,12)
    assert coord_xyz[0, 0] == "C"
    assert float(coord_xyz[1, 0]) == -2.97431
    assert coord_xyz[0, 6] == "H"
    coord_xyz, header = top.read_xyz_file("./tests/test_data/benz.xyz", return_header=True)
    assert header == "Header eins"

def test_read_hessian():
    #"""
    hessians = ["hessian_tm", "hessian_xtb"]
    for hessian in hessians:
        with pytest.raises(ValueError):
            top.read_hessian(f"./tests/test_data/{hessian}", n_atoms=1, dimensions=3)
        with pytest.raises(ValueError):
            top.read_hessian(f"./tests/test_data/{hessian}", n_atoms=3, dimensions=3)

        hessian = top.read_hessian(f"./tests/test_data/{hessian}", n_atoms=2, dimensions=3)
        assert np.max(np.abs(hessian-np.transpose(hessian))) == 0
        assert hessian[0,0]

def test_read_symmetric_from_triangular():
    hessian = top.read_symmetric_from_triangular("./tests/test_data/hessian_direct")
    #symmetry
    assert np.max(hessian-np.transpose(hessian)) == 0
    assert hessian.shape == (3*3, 3*3)
    assert hessian[0,0] == 3.806638367204165E-002
    assert hessian[8,8] == 0.235824323786876
    assert hessian[7,8] == -0.222050038511708
    assert hessian[6,8] == -0.112940116315714
    top.write_symmetric_to_triangular(hessian, "/tmp/hessian")
    hessian_reread = top.read_symmetric_from_triangular("/tmp/hessian")
    assert np.max(np.abs(hessian_reread-hessian)) == 0

def test_read_from_flag_to_flag():
    control_path = "./tests/test_data/control_prj"
    status = top.read_from_flag_to_flag(control_path, "$notvalid", "/tmp/flag_output_1")
    assert status == -1
    assert os.path.exists("/tmp/flag_output_1") == False

    status = top.read_from_flag_to_flag(control_path, "$nosalc", "/tmp/flag_output_2")
    assert status == -2
    assert os.path.exists("/tmp/flag_output_2") == False

    status = top.read_from_flag_to_flag(control_path, "$nprhessian", "/tmp/flag_output.dat")
    assert status == 0
    assert os.path.exists("/tmp/flag_output.dat") == True


if __name__ == '__main__':
    test_load_xyz_file()
    test_determine_n_orbitals()