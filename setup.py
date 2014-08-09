from distutils.core import setup
setup(name='pymvg',
      version='1.1',
      packages=['pymvg',
                'pymvg.test',
                'pymvg.test.external.opencv',
                'pymvg.test.external.ros',
                'pymvg.test.external.mcsc',
                ],
      url='https://github.com/strawlab/pymvg',
      package_data={ 'pymvg.test': ['*.yaml'],
                     'pymvg.test': ['*.json'],
                     },
      )
