from setuptools import setup, find_packages

def load_requirements(filename):
    with open(filename, 'r') as f:
        return f.read().splitlines()

setup(
    name='digitaltwins',
    version='0.1',
    packages=find_packages(),
    install_requires=load_requirements('requirements.txt'),
    extras_require={
        'gpu': load_requirements('requirements_gpu.txt'),
        'cpu': load_requirements('requirements_cpu.txt')

    },
    entry_points={
        'console_scripts': [
            'run-main=digitaltwins.main:main',
            'run-sim=digitaltwins.sim:main',
            'run-eval=digitaltwins.eval:main',
            'run-plot=digitaltwins.plot_quantiles:main',
            'run-summary-sim=digitaltwins.summary_sim:main',

        ],
    },
    description='Digital Twins package, which simulates data and estimates the model',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Samuel Levy',
    author_email='silevy@andrew.cmu.edu',
    url='https://github.com/silevy/digitaltwins',
)
