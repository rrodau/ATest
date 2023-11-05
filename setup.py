with open("README.md", "r") as fh:
    long_description = fh.read()

from distutils.core import setup

setup(name='FairRanking',
      version='0.1',
      description='Selection of neuronal network based FairRankers',
      author="No Author Given",
      author_email="no@author.given",
      long_description=long_description,
      long_description_content_type="text/markdown",
      packages=['FairRanking', 'FairRanking.datasets'],
      )
