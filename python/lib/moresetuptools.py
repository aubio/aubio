import distutils, os, subprocess
import setuptools.command.build_py
import distutils.command.clean
from distutils.dir_util import remove_tree

class CleanGenerated(distutils.command.clean.clean):
    def run(self):
        remove_tree('gen')
        distutils.command.clean.clean.run(self)

class GenerateCommand(distutils.cmd.Command):
    description = 'generate gen/gen-*.c files from ../src/aubio.h'
    user_options = [
            # The format is (long option, short option, description).
            ('enable-double', None, 'use HAVE_AUBIO_DOUBLE=1 (default: 0)'),
            ]

    def initialize_options(self):
        self.enable_double = False

    def finalize_options(self):
        if self.enable_double:
            self.announce(
                    'will generate code for aubio compiled with HAVE_AUBIO_DOUBLE=1',
                    level=distutils.log.INFO)

    def run(self):
        self.announce( 'Generating code', level=distutils.log.INFO)
        from .gen_external import generate_external
        generated_object_files = generate_external('gen', usedouble = self.enable_double)
