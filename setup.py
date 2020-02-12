from setuptools import find_packages, setup

setup(
    name='mlpy',
    version='0.1',
    description="Minimalist Library for Machine Learning Project",
    url='https://github.com/kcosta42/Multilayer_Perceptron',
    author='kcosta',
    author_email='kcosta@student.42.fr',
    license='MIT',
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=['numpy', 'matplotlib', 'pandas'],
)
