import os.path
import sys


path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path)

os.system("pdoc tmoutproc -o docs")