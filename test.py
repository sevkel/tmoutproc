import os

if __name__ == '__main__':
    try:
        os.system("pdoc ./tmoutproc -o docs/")
    except RuntimeError as e:
        print(e)