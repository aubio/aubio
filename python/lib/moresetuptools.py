import distutils, distutils.command.clean, distutils.dir_util
from .gen_external import generate_external, header, output_path

class CleanGenerated(distutils.command.clean.clean):
    def run(self):
        distutils.dir_util.remove_tree(output_path)
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
        generated_object_files = generate_external(header, output_path, usedouble = self.enable_double)
