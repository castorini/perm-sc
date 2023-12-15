import setuptools


setuptools.setup(
    name='permsc',
    version=eval(open('permsc/_version.py').read().strip().split('=')[1]),
    author='Blinded',
    license='MIT',
    url='https://blinded.ai',
    author_email='blinded@blinded.ai',
    description='Official codebase for permutation self-consistency.',
    install_requires=open('requirements.txt').read().strip().splitlines(),
    packages=setuptools.find_packages(),
    python_requires='>=3.10',
)
