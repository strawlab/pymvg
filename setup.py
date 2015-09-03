from distutils.core import setup, Command

class NoseTestCommand(Command):
    user_options = [ ]

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import nose
        nose.run_exit(argv=['nosetests'])

import pymvg as this

setup(name='pymvg',
      description = 'Python Multi-View Geometry',
      author='Andrew Straw',
      author_email='strawman@astraw.com',
      version=this.__version__,
      packages=['pymvg',
                'pymvg.extern',
                'pymvg.extern.opencv',
                'pymvg.extern.ros',
                'pymvg.test',
                'pymvg.test.external.opencv',
                'pymvg.test.external.ros',
                'pymvg.test.external.mcsc',
                ],
      url='https://github.com/strawlab/pymvg',
      package_data={ 'pymvg.test': ['*.yaml','*.json'],
                     },
      cmdclass={'test': NoseTestCommand},
      )
