from turbomoleOutputProcessing import turbomoleOutputProcessing as top
def test_load_xyz_file():
    #problem not acutal code is used but installed code
    header, coord_xyz = top.load_xyz_file("./tests/test_data/benz.xyz")

    assert header[0] == "Header"
    assert coord_xyz.shape == (4,12)
    assert coord_xyz[0, 0] == "C"
    assert float(coord_xyz[1, 0]) == -2.97431
    assert coord_xyz[0, 6] == "H"


if __name__ == '__main__':
    test_load_xyz_file()