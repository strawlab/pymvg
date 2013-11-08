from distutils.core import setup
setup(name='pymvg',
      version='1.0',
      packages=['pymvg',
                'pymvg.test',
                'pymvg.test.external.opencv',
                'pymvg.test.external.ros',
                'pymvg.test.external.mcsc',
                ],
      package_data={ 'pymvg.test': ['*.yaml'],
                     },
      )
