from setuptools import setup

setup(
  name = 'tensorflowonspark',
  packages = ['tensorflowonspark'],
  version = '1.4.0',
  description = 'Deep learning with TensorFlow on Apache Spark clusters',
  author = 'Yahoo, Inc.',
  url = 'https://github.com/yahoo/TensorFlowOnSpark',
  keywords = ['tensorflowonspark', 'tensorflow', 'spark', 'machine learning', 'yahoo'],
  install_requires = ['tensorflow'],
  license = 'Apache 2.0',
  classifiers = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: Apache Software License',
    'Topic :: Software Development :: Libraries',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6'
  ]
)
