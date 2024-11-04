import tmoutproc as top
import pytest
import numpy as np
import os.path
import filecmp
import scipy.sparse

def test_read_write_xyz_file():
    #problem not acutal code is used but installed code
    coord_xyz = top.read_xyz_file("./tests/test_data/benz.xyz")

    assert coord_xyz.shape == (4,12)
    assert coord_xyz[0, 0] == "C"
    assert float(coord_xyz[1, 0]) == -2.97431
    assert coord_xyz[0, 6] == "H"
    coord_xyz, header = top.read_xyz_file("./tests/test_data/benz.xyz", return_header=True)
    assert header == "Header eins"

    top.write_xyz_file("/tmp/xyz.xyz", coord_xyz, "test")
    coord_xyz_reread, header = top.read_xyz_file("/tmp/xyz.xyz", return_header=True)
    assert header == "test"
    assert coord_xyz.shape == coord_xyz_reread.shape
    assert np.max(np.abs(coord_xyz[1:3,:]-coord_xyz_reread[1:3,:])) == 0

def test_read_write_coord_file():

    coord_PCP = top.read_coord_file("./tests/test_data/coord_PCP")

    assert coord_PCP.shape == (5,90)

    assert coord_PCP[0,20] == -0.56571033834128
    assert coord_PCP[1,20] == 0.04245149989675
    assert coord_PCP[2,20] == 2.16496571320022
    assert coord_PCP[3,20] == "c"
    assert coord_PCP[4, 20] == ""

    assert coord_PCP[0, 89] == -4.84905613305097
    assert coord_PCP[1, 89] == -0.90619926580304
    assert coord_PCP[2, 89] == 39.80494546134004
    assert coord_PCP[3, 89] == "au"
    assert coord_PCP[4, 89] == "f"

    top.write_coord_file("/tmp/coord_pcp", coord_PCP)
    coord_PCP = top.read_coord_file("/tmp/coord_pcp")

    assert coord_PCP[0,20] == -0.56571033834128
    assert coord_PCP[1,20] == 0.04245149989675
    assert coord_PCP[2,20] == 2.16496571320022
    assert coord_PCP[3,20] == "c"
    assert coord_PCP[4, 20] == ""

    assert coord_PCP[0, 89] == -4.84905613305097
    assert coord_PCP[1, 89] == -0.90619926580304
    assert coord_PCP[2, 89] == 39.80494546134004
    assert coord_PCP[3, 89] == "au"
    assert coord_PCP[4, 89] == "f"


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

def test_create_sysinfo():
    test_data_path = "./tests/test_data"
    top.create_sysinfo(f"{test_data_path}/coord_sysinfo", f"{test_data_path}/basis_sysinfo", "/tmp/sysinfo.dat")
    assert filecmp.cmp(f"{test_data_path}/sysinfo.dat", "/tmp/sysinfo.dat") == True

def test_read_write_matrix_packed():
    test_data_path = "./tests/test_data"
    matrix = top.read_packed_matrix(f"{test_data_path}/fmat_ao_cs_benz.dat")
    assert type(matrix) == scipy.sparse.csc_matrix
    matrix_dense = top.read_packed_matrix(f"{test_data_path}/fmat_ao_cs_benz.dat", output="dense")
    with pytest.raises(ValueError):
        top.read_packed_matrix(f"{test_data_path}/fmat_ao_cs_benz.dat", output="notvalid")
    assert np.max(np.abs(matrix.todense()-matrix_dense)) == 0
    assert matrix_dense[0,0] == -9.885587065525669459020719
    n = int((-1+np.sqrt(1+8*4656))/2)
    assert matrix_dense.shape == (n,n)
    assert matrix_dense[n-1, n-1] == -0.3948151940509845303495240
    assert np.max(matrix_dense-np.transpose(matrix_dense)) == 0

    top.write_packed_matrix(matrix_dense, "/tmp/packed_matrix_dense")
    re_read_dense = top.read_packed_matrix("/tmp/packed_matrix_dense", "dense")
    assert np.max(np.abs(re_read_dense - matrix_dense)) == 0

    top.write_packed_matrix(matrix, "/tmp/packed_matrix")
    re_read_dense = top.read_packed_matrix("/tmp/packed_matrix")
    assert np.max(np.abs(re_read_dense - matrix)) == 0

def test_read_mos_file():
    eigenvalues, eigenvectors = top.read_mos_file("./tests/test_data/mos_benz")
    assert len(eigenvalues) == 96
    print(eigenvalues)
    assert  eigenvalues[0] == -.98930871549516E+01
    assert eigenvalues[95] == 0.32978958538886E+01
    assert  eigenvectors[0, 0] == 0.40333702240522
    assert eigenvectors[1, 0] == 0.16149860838108E-01
    assert eigenvectors[2, 0] == -0.68722499925093E-02

    assert eigenvectors[0, 95] == 0.61752896138301E-01
    assert eigenvectors[1, 95] == -.23316991833123E+00
    assert eigenvectors[2, 95] == -.42986018618970E+01

    top.write_mos_file(eigenvalues, eigenvectors, "/tmp/test_mos")
    eigenvalues_reread, eigenvectors_reread = top.read_mos_file("/tmp/test_mos")
    assert np.max(np.abs(np.array(eigenvalues_reread)-np.array(eigenvalues))) == 0
    assert np.max(np.abs(eigenvectors_reread - eigenvectors)) == 0

    eigenvalues, eigenvectors = top.read_mos_file("./tests/test_data/mos_not_div_by_4")
    assert type(eigenvectors) == np.ndarray


    assert eigenvalues[0] == -.18826938211009E+02
    assert eigenvalues[45] == 0.36814130449147E+01
    assert eigenvectors.shape == (46,46)
    assert eigenvectors[44,45] == 0.49650335243185E-01

def test_read_write_plot_data():
    array1 = np.linspace(0, 1, 100)
    array2 = np.linspace(4, 3, 100)
    array3 = np.linspace(5, 8, 100)

    top.write_plot_data("/tmp/plot_data.dat", [array1, array2, array3], header="test")
    data = top.read_plot_data("/tmp/plot_data.dat", False)
    data, header = top.read_plot_data("/tmp/plot_data.dat", True)
    assert header == "test"
    assert np.max(np.abs(data[0,:]-array1)) == 0
    assert np.max(np.abs(data[1, :] - array2)) == 0

    top.write_plot_data("/tmp/plot_data.dat", (array1, array2, array3), header="test")
    data, header = top.read_plot_data("/tmp/plot_data.dat", True)
    assert header == "test"
    assert np.max(np.abs(data[0, :] - array1)) == 0
    assert np.max(np.abs(data[1, :] - array2)) == 0

    combined = np.vstack((array1, array2, array3))
    print(type(combined))
    top.write_plot_data("/tmp/plot_data.dat", combined, header="test")
    data, header = top.read_plot_data("/tmp/plot_data.dat", True)
    assert header == "test"
    assert np.max(np.abs(data[0, :] - array1)) == 0
    assert np.max(np.abs(data[1, :] - array2)) == 0

    with pytest.raises(ValueError):
        top.write_plot_data("/tmp/plot_data.dat", "not valid", header="test", delimiter="")
    with pytest.raises(ValueError):
        top.write_plot_data("/tmp/plot_data.dat", [array1, array2, array3], header="test", delimiter="")
    with pytest.raises(ValueError):
        top.write_plot_data("/tmp/plot_data.dat", [array1, array2, array3], header="test", delimiter="3")

    top.write_plot_data("/tmp/plot_data.dat", [array1, array2, array3], header="test", delimiter=",")
    data, header = top.read_plot_data("/tmp/plot_data.dat", True, delimiter=",")
    assert header == "test"
    assert np.max(np.abs(data[0, :] - array1)) == 0
    assert np.max(np.abs(data[1, :] - array2)) == 0
    with pytest.raises(ValueError):
        top.read_plot_data("/tmp/plot_data.dat", True, delimiter=".")

    with pytest.raises(ValueError):
        top.read_plot_data("./tests/test_data/plot_data.dat", True, delimiter=",")

    data = top.read_plot_data("./tests/test_data/plot_data.dat", False, delimiter=",", skip_lines_beginning=2)
    assert data.shape == (7, 7)
    compare = [300, 301, 302, 303, 304, 305, 306]
    assert np.all([data[0,i] == compare[i] for i in range(0,data.shape[0])])
    data = top.read_plot_data("./tests/test_data/plot_data.dat", False, delimiter=",", skip_lines_beginning=2, skip_lines_end=2)
    assert data.shape == (7, 7-2)


def test_read_g98_file():
    modes = top.read_g98_file("./tests/test_data/g98_test.g98")

    assert len(modes) == 5
    freqs = []
    red_masses = []
    for mode in modes:
        keys = mode.keys()
        print(keys)
        assert "coord_xyz" in keys
        assert mode["coord_xyz"].shape == (4, 38)
        assert "mode_xyz" in keys
        assert mode["mode_xyz"].shape == (4, 38)
        assert "frequency" in keys
        freqs.append(mode["frequency"])
        assert "red_mass" in keys
        red_masses.append(mode["red_mass"])
        assert "frc_const" in keys
        assert "ir_intensity" in keys
        assert "raman_activity" in keys
        assert "depolarization_ratio" in keys

    mode_xyz = modes[0]["mode_xyz"]
    assert mode_xyz[0, 0] == "S"
    ref = [-0.07,   0.04,  -0.00]
    assert np.max(np.abs(np.array(mode_xyz[1:, 1]) - np.array(ref))) < 1e-12

    ref_freqs = [11.7111, 13.8197, 14.1691, 32.4986, 32.6780]
    assert np.max(np.abs(np.array(freqs)-np.array(ref_freqs))) < 1e-4
    ref_red_masses = [20.7570, 10.4306, 10.0481,15.4042, 17.8718]
    assert np.max(np.abs(np.array(red_masses)-np.array(ref_red_masses))) < 1e-4


    with pytest.raises(AssertionError):
        modes = top.read_g98_file("./tests/test_data/g98_test_faulty1.g98")

    with pytest.raises(AssertionError):
        modes = top.read_g98_file("./tests/test_data/g98_test_faulty2.g98")


@pytest.fixture
def setup_test_file():
    """Fixture to set up and tear down the test file."""
    filename = "./tests/test_data/test_nmd.nmd"
    modes = [
        {
            "coord_xyz": np.array([["H", "O", "H"], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 2.0]]),
            "mode_xyz": np.array([["H", "O", "H"], [0, 0, 0], [0.1, 0, 0], [-0.1, 0, 4.0]])
        }
    ]
    yield filename, modes
    if os.path.exists(filename):
        os.remove(filename)


def test_write_nmd_file(setup_test_file):
    """Test writing to NMD file."""
    filename, modes = setup_test_file
    top.write_nmd_file(filename, modes, scale_with_mass=False)

    # Check if the file was created
    assert os.path.exists(filename)

    with open(filename, 'r') as file:
        lines = file.readlines()

    # Check the content of the file
    assert lines[0].strip() == "names H O H"
    assert lines[1].strip() == "coordinates 0.0 0.0 0.0 0.0 1.0 1.0 0.0 0.0 2.0"
    assert lines[2].strip() == "mode 0.0 0.1 -0.1 0.0 0.0 0.0 0.0 0.0 4.0"


def test_read_nmd_file_invalid_format(setup_test_file):
    """Test reading from an invalid NMD file."""
    filename, _ = setup_test_file
    with open(filename, 'w') as file:
        file.write("invalid content\n")

    read_modes = top.read_nmd_file(filename)

    # Expecting no modes to be read due to invalid format
    assert read_modes == []


def test_read_nmd_file(setup_test_file):
    """Test reading from NMD file."""
    filename, modes = setup_test_file
    top.write_nmd_file(filename, modes, scale_with_mass=False)

    read_modes = top.read_nmd_file(filename)

    # Check the content read from the file
    assert len(read_modes) == 1
    assert read_modes[0]["mode_xyz"].shape == modes[0]["mode_xyz"].shape
    print(read_modes[0]["coord_xyz"][1:,], modes[0]["coord_xyz"][1:,])
    assert np.array_equal(read_modes[0]["coord_xyz"], modes[0]["coord_xyz"])









