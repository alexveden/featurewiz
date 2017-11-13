from setuptools import setup

setup(name='featurewiz',
      version='0.1a',
      description='Data science features toolbox',
      url='https://github.com/alexveden/featurewiz',
      author='Alex Veden',
      author_email='i@alexveden.com',
      license='MIT',
      packages=['funniest'],
      install_requires=[
          'pandas',
      ],
      zip_safe=False)