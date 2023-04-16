from setuptools import setup, find_packages


setup(name='gcn',
      version='1.0.0',
      description='Semi-supervised GCN paper\'s implementation in pytorch.',
      license="MIT",
      author='Raihan Islam Arnob',
      author_email='rarnob@gmu.edu',
      packages=find_packages(),
      install_requires=['numpy', 'matplotlib'])
