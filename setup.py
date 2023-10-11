import setuptools


setuptools.setup(
    name='permsc',
    version=eval(open('permsc/_version.py').read().strip().split('=')[1]),
    author='Raphael Tang',
    license='MIT',
    url='https://github.com/castorini/perm-sc',
    author_email='r33tang@uwaterloo.ca',
    description='Official codebase for permutation self-consistency.',
    install_requires=open('requirements.txt').read().strip().splitlines(),
    packages=setuptools.find_packages(),
    python_requires='>=3.10',
)
