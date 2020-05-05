from setuptools import setup

setup(name='mplt',
      version='2.1.0',
      description='Set of niceties wrapping matplotlib for signal processing',
      url='http://github.com/stringhamc/mplt',
      author='Craig Stringham',
      author_email='stringham@ieee.org',
      license='MIT',
      packages=['mplt'],
      classifiers=[
          # Indicate who your project is intended for
          'Intended Audience :: Science/Research',
      ],
      install_requires=['matplotlib', 'numpy']
)
