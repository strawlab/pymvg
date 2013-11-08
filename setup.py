from distutils.core import setup
setup(name='pymvg',
      version='1.0',
      packages=['pymvg',
                'pymvg.test',
                'pymvg.test.external.cv',
                'pymvg.test.external.ros',
                'pymvg.test.external.mcsc',
                ],
      )
