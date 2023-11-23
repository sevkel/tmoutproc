import tmoutproc as top
import pytest
import numpy as np


def test_shift_xyz_coord():
    coord_xyz = top.read_xyz_file("./tests/test_data/benz.xyz")
    coord = top.read_coord_file("./tests/test_data/coord_PCP")

    coord_xyz_shifted = top.shift_xyz_coord(coord_xyz, 1.0 ,2.0 ,3.0)
    assert np.isclose(float(coord_xyz_shifted[1, 0]),-2.97431+1.0)
    assert np.isclose(float(coord_xyz_shifted[2, 0]),0.42856 + 2.0)
    assert np.isclose(float(coord_xyz_shifted[3, 0]),-0.00000 + 3.0)

    assert np.isclose(float(coord_xyz_shifted[1, 11]), -4.08087  + 1.0)
    assert np.isclose(float(coord_xyz_shifted[2, 11]), -2.79039 + 2.0)
    assert np.isclose(float(coord_xyz_shifted[3, 11]), -0.00000 + 3.0)


def test_shift_coord_file():

    coord = top.read_coord_file("./tests/test_data/coord_PCP")

    coord_shifted = top.shift_coord_file(coord, 1.0 ,2.0 ,3.0)
    assert type(coord_shifted[0, 0] == float)
    assert np.isclose(coord_shifted[0, 0],-8.17778980431802+1.0)
    assert np.isclose(coord_shifted[1, 0],-4.72144292785499 + 2.0)
    assert np.isclose(coord_shifted[2, 0],-12.84405276595290 + 3.0)

    assert np.isclose(coord_shifted[0, 89], -4.84905613305097  + 1.0)
    assert np.isclose(coord_shifted[1, 89], -0.90619926580304 + 2.0)
    assert np.isclose(coord_shifted[2, 89], 39.80494546134004 + 3.0)

def test_sort_xyz_coord():
    coord_xyz = top.read_xyz_file("./tests/test_data/benz.xyz")
    with pytest.raises(ValueError):
        top.sort_xyz_coord(coord_xyz, axis=3)
    axes = [2,1,0]
    # test sorting
    for axis in axes:
        coord_sorted = top.sort_xyz_coord(coord_xyz, axis=axis)
        assert coord_sorted.shape == coord_xyz.shape
        assert all(coord_sorted[axis+1, i] <= coord_sorted[axis+1, i+1] for i in range(len(coord_sorted[:, axis+1]) - 1))
    # test if elements are correct
    assert coord_sorted[0, 0] == "H"
    assert coord_sorted[1, 0] == -6.35080
    assert coord_sorted[2, 0] == 0.85915
    assert coord_sorted[3, 0] == 0.00000

def test_sort_coord():
    coord = top.read_coord_file("./tests/test_data/coord_PCP")
    with pytest.raises(ValueError):
        top.sort_xyz_coord(coord, axis=3)
    axes = [2,0,1]
    for axis in axes:
        coord_sorted = top.sort_coord(coord, axis)
        assert coord_sorted.shape[1] == coord.shape[1]
        assert all(coord_sorted[axis,i] <= coord_sorted[axis,i+1] for i in range(coord_sorted.shape[1] - 1))

    assert coord_sorted[0, 0] == 0.60280373649437
    assert coord_sorted[1, 0] == -10.34910401877428
    assert coord_sorted[2, 0] == 39.80494546134004
    assert coord_sorted[3, 0] == 'au'
    assert coord_sorted[4, 0] == 'f'

    with pytest.raises(ValueError):
        coord_sorted = top.sort_coord(coord, 42)

def test_x2t():
    coord_xyz = top.read_xyz_file("./tests/test_data/coord_PCP.xyz")
    n_atoms = coord_xyz.shape[1]
    coord = top.read_coord_file("./tests/test_data/coord_PCP")
    coord_generated = top.x2t(coord_xyz)
    assert coord_generated.shape == (5, n_atoms)

    thresh = 1E-4
    diff = [coord[0, i] - coord_generated[0, i] for i in range(0, coord.shape[1])]
    assert np.max(np.max(diff)) < thresh
    diff = [coord[1, i] - coord_generated[1, i] for i in range(0, coord.shape[1])]
    assert np.max(np.max(diff)) < thresh
    diff = [coord[2, i] - coord_generated[2, i] for i in range(0, coord.shape[1])]
    assert np.max(np.max(diff)) < thresh

    is_same = [coord[3, i] == coord_generated[3, i] for i in range(0, coord.shape[1])]
    assert np.all(is_same)

def test_t2x():
    coord_xyz = top.read_xyz_file("./tests/test_data/coord_PCP.xyz")
    n_atoms = coord_xyz.shape[1]
    coord = top.read_coord_file("./tests/test_data/coord_PCP")
    coord_xyz_generated = top.t2x(coord)
    assert coord_xyz_generated.shape == (4, n_atoms)

    thresh = 1E-4
    diff = [coord_xyz_generated[3, i] - coord_xyz[3, i] for i in range(0, coord_xyz.shape[1])]
    assert np.max(np.max(diff)) < thresh
    diff = [coord_xyz_generated[1, i] - coord_xyz[1, i] for i in range(0, coord_xyz.shape[1])]
    assert np.max(np.max(diff)) < thresh
    diff = [coord_xyz_generated[2, i] - coord_xyz[2, i] for i in range(0, coord_xyz.shape[1])]
    assert np.max(np.max(diff)) < thresh

    is_same = [coord_xyz_generated[0, i] == coord_xyz[0, i] for i in range(0, coord_xyz.shape[1])]
    assert np.all(is_same)


def test_fix_atoms():
    coord = top.read_coord_file("./tests/test_data/coord_PCP_filtered")
    coord_fixed = top.fix_atoms(coord, "all")

    assert coord.shape[1] == coord_fixed.shape[1]

    is_fixed = [coord_fixed[4,i] == "f" for i in range(0, coord_fixed.shape[1])]
    assert np.all(is_fixed)

    coord = top.read_coord_file("./tests/test_data/coord_PCP_filtered")
    coord_fixed = top.fix_atoms(coord, [1, 10, 20])
    is_fixed = [coord_fixed[4,i]== "f" for i in [1, 10, 20]]
    is_not_fixed = [coord_fixed[4,i] == "" for i in [2, 3, 4]]
    assert np.all(is_fixed)
    assert np.all(is_not_fixed)

    coord = top.read_coord_file("./tests/test_data/coord_h2")
    indices = np.arange(0,2)
    coord_fixed = top.fix_atoms(coord, indices)
    is_fixed = [coord_fixed[4, i] == "f" for i in range(0, coord_fixed.shape[1])]
    assert np.all(is_fixed)

    coord = top.read_coord_file("./tests/test_data/coord_sysinfo")
    coord_unfixed = top.fix_atoms(coord, [], unfix_first=True)
    is_not_fixed = [coord_unfixed[4, i] == "" for i in [2, 3, 4]]
    assert np.all(is_not_fixed)


    coord = top.read_coord_file("./tests/test_data/coord_sysinfo")
    coord = top.fix_atoms(coord, [], unfix_first=False)
    assert coord[4,0] == "f"

    #test "invert" option
    coord = top.read_coord_file("./tests/test_data/coord_h2")
    coord_fixed = top.fix_atoms(coord, "invert")
    is_fixed = [coord_fixed[4, i] == "f" for i in range(0,coord_fixed.shape[1])]
    assert np.all(is_fixed)

    coord = top.read_coord_file("./tests/test_data/coord_h2")
    coord_fixed = top.fix_atoms(coord, 1, unfix_first=True)
    assert coord_fixed[4, 0] == ""
    assert coord_fixed[4, 1] == "f"

    coord_fixed = top.fix_atoms(coord_fixed, "invert", unfix_first=False)
    assert coord_fixed[4, 0] == "f"
    assert coord_fixed[4, 1] == ""

    #test input handling

    with pytest.raises(ValueError):
        top.fix_atoms(coord, "NOT VALID")
    with pytest.raises(ValueError):
        top.fix_atoms(coord, [1,2.2,3])






def test_remove_fixed_atoms_from_coord():
    coord = top.read_coord_file("./tests/test_data/coord_PCP")
    ref_len = coord.shape[1]
    coord_PCP_filtered = top.remove_fixed_atoms_from_coord(coord)
    assert coord.shape[1] == 90, "Error in read_coord_file"
    assert coord_PCP_filtered.shape == (5,ref_len - 16 * 2)
    assert coord_PCP_filtered[3,0] == 'au', "This element must be au"
    assert coord_PCP_filtered[0,0] == -0.03330929466830, "Wrong value in coord_PCP_filtered"


def test_align_molecule():
    coord_xyz = top.read_xyz_file("./tests/test_data/coord_simple.xyz")
    coord_xyz_rotated = top.align_molecule(coord_xyz, [1,1,1], [0,1])

    ref_dist = np.linalg.norm(coord_xyz[1:4, 0] - coord_xyz[1:4, 1])
    rotated_dist = np.linalg.norm(coord_xyz_rotated[1:4, 0] - coord_xyz_rotated[1:4, 1])
    assert np.isclose(rotated_dist, ref_dist)

    assert np.isclose(coord_xyz[1,1],coord_xyz[2,1]) and  np.isclose(coord_xyz[2,1],coord_xyz[3,1])

    coord_xyz = top.read_xyz_file("./tests/test_data/rotation_test.xyz")
    coord_xyz_rotated = top.align_molecule(coord_xyz, [0, 0, 1], [1, 0])
    assert  coord_xyz.shape == coord_xyz_rotated.shape
    is_same = [np.isclose(coord_xyz[1,i], coord_xyz_rotated[1,i]) for i in range(0,coord_xyz_rotated.shape[1])]
    assert np.all(is_same), "x coord changed"
    is_same = [np.isclose(coord_xyz[2, i], coord_xyz_rotated[2, i]) for i in range(0, coord_xyz_rotated.shape[1])]
    assert np.all(is_same), "y coord changed"
    is_same = [np.isclose(coord_xyz[3, i], coord_xyz_rotated[3, i]) for i in range(0, coord_xyz_rotated.shape[1])]
    assert np.all(is_same), "z coord changed"
    is_same = [coord_xyz[0, i] == coord_xyz_rotated[0, i] for i in range(0, coord_xyz_rotated.shape[1])]
    assert np.all(is_same), "atom changed"

    coord_xyz = top.read_xyz_file("./tests/test_data/rotation_test.xyz")
    coord_xyz_rotated = top.align_molecule(coord_xyz, [1, 0, 0], [1, 0])
    assert np.isclose(coord_xyz_rotated[1, 0], 7.16848)
    assert np.isclose(coord_xyz_rotated[2, 0], 0)
    assert np.isclose(coord_xyz_rotated[3, 0], 0)
    assert np.isclose(coord_xyz_rotated[1, 1], 0)
    assert np.isclose(coord_xyz_rotated[2, 1], 0)
    assert np.isclose(coord_xyz_rotated[3, 1], 0)
    assert np.isclose(coord_xyz_rotated[1, 20], 5.85707)


    ref_dist = [np.linalg.norm(coord_xyz[1:4, i] - coord_xyz[1:4, i+1]) for i in range(coord_xyz.shape[0]-1)]
    rotated_dist =[np.linalg.norm(coord_xyz_rotated[1:4, i] - coord_xyz_rotated[1:4, i+1]) for i in range(coord_xyz.shape[0]-1)]
    is_close = [np.isclose(ref_dist[i], rotated_dist[i]) for i in range(len(rotated_dist))]
    assert np.all(is_close), "Distance changed"

