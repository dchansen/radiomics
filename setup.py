from distutils.core import setup

setup(
    name='radiomics',
    version='0.1',
    packages=['radiomics'],
    url='https://github.com/dchansen/radiomics',
    license='Apache',
    author='dch',
    author_email='daviha@rm.dk',
    description='A radiomics implementation in python',
    install_requires=['numpy', 'scipy', 'scikit-image', 'PyWavelets']
)
