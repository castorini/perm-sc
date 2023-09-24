import setuptools


setuptools.setup(
    name='fastrank',
    version=eval(open('fastrank/_version.py').read().strip().split('=')[1]),
    author='Raphael Tang',
    license='MIT',
    url='https://github.com/castorini/fastrank',
    author_email='r33tang@uwaterloo.ca',
    description='FastRank: A fast rank aggregation library.',
    install_requires=open('requirements.txt').read().strip().splitlines(),
    packages=setuptools.find_packages(),
    python_requires='>=3.10',
    extras_require={
        'llm': [
            'beir==2.0.0'
        ]
    }
)
