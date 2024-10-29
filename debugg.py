import numpy as np
import fnmatch
import scipy.sparse
import re as r
from scipy.linalg import eig
from functools import partial
from multiprocessing import Pool
from scipy.sparse import coo_matrix


def read_g98_file(filepath):

    #read coord part



    with open(filepath) as file:
        lines = file.readlines()

        modes = {}

        modepart_found = False
        coord_part_found = False
        part_length = -1
        modes_parts = []
        frequencies = []
        for i,line in enumerate(lines):
            #coords are specified (wildcards before and after)
            if "coordinates" in line.lower() and "atomic" in line.lower():
                coord_part_found = True
                continue


            #Frequncies are specified
            if r.match(r"\s*Frequencies", line):
                part = line.strip().split()
                for item in part:
                    try:
                        frequency = float(item)
                        frequencies.append(frequency)
                    except ValueError:
                        pass

            #Description of modes starts
            if r.match(r"\s*Atom", line) and len(frequencies) > 0:
                modepart_found = True
                continue
            if modepart_found:
                part = line.strip().split()
                #mode part just starts -> implementation expects mode part longer than one atom
                if(part_length == -1):
                    part_length = len(part)

                #mode part is over
                if(part_length != len(part)):
                    if len(part) != part_length:
                        mode_parts = np.array(modes_parts, dtype="object")
                        n_modes = int((mode_parts.shape[1] - 2) / 3)
                        assert n_modes == len(frequencies)
                        for mode in range(n_modes):
                            print(mode)
                            mode_xyz = mode_parts[:,2+mode*3:2+mode*3+3]
                            #add atom type column to xyz
                            mode_xyz = np.column_stack((mode_xyz, mode_parts[:,1]))
                            mode_xyz = mode_xyz.T
                            modes[frequencies[mode]] = mode_xyz
                        modes_parts = []
                        frequencies = []
                        modepart_found = False

                else:
                    modes_parts.append(part)
        print(modes)







if __name__ == '__main__':
    #read_mos_file("../tests/test_data/mos_benz")
    read_g98_file("./tests/test_data/g98_test.txt")