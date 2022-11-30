from setuptools import setup

setup(
    name='tmoutproc',
    url='https://github.com/blaschma/tmoutproc',
    author='Matthias Blaschke',
    author_email='matthias.blaschke@student.uni-augsburg.de',
    packages=['tmoutproc'],
    install_requires=['numpy', 'scipy', 're', 'functools'],
    version='0.1',
    license='MIT',
    description='Package for processing turbomole Output',
)
