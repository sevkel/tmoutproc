from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='turbomoleOuptputProcessing',
    url='https://github.com/jladan/package_demo',
    author='Matthias Blaschke',
    author_email='matthias.blaschke@student.uni-augsburg.de',
    # Needed to actually package something
    packages=['turbomoleOutputProcessing'],
    # Needed for dependencies
    install_requires=['numpy', 'scipy','numba'],
    # *strongly* suggested for sharing
    version='0.1',
    # The license can be anything you like
    license='MIT',
    description='Package for processing turbomole Output such as mos files',
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)
