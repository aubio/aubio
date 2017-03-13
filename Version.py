
import os
for l in open('VERSION').readlines(): exec (l.strip())

# def get_git_revision_hash( short=True):
#     import os
#     def which(program):
#         def is_exe(fpath):
#             return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

#         fpath, fname = os.path.split(program)
#         if fpath:
#             if is_exe(program):
#                 return program
#         else:
#             for path in os.environ["PATH"].split(os.pathsep):
#                 path = path.strip('"')
#                 exe_file = os.path.join(path, program)
#                 if is_exe(exe_file):
#                     return exe_file
#         return None

#     if not which('git') :
#         # print('no git found on this system : can\'t get sha')
#         return ""
#     if not os.path.isdir('.git'):
#         # print('Version : not in git repository : can\'t get sha')
#         return ""

#     import subprocess
#     aubio_dir = os.path.abspath(os.curdir)
#     if not os.path.exists(aubio_dir):
#         raise SystemError("git / root folder not found")
#     gitcmd = ['git','-C',aubio_dir ,'rev-parse']
#     if short:
#       gitcmd.append('--short')
#     gitcmd.append('HEAD')
#     outCmd = subprocess.check_output(gitcmd).strip().decode('utf8')
#     return outCmd


def get_aubio_version():
    # read from VERSION
    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    version_file = os.path.join(this_file_dir,  'VERSION')

    if not os.path.isfile(version_file):
        raise SystemError("VERSION file not found.")

    for l in open(version_file).readlines():
        #exec (l.strip())
        if l.startswith('AUBIO_MAJOR_VERSION'):
            AUBIO_MAJOR_VERSION = int(l.split('=')[1])
        if l.startswith('AUBIO_MINOR_VERSION'):
            AUBIO_MINOR_VERSION = int(l.split('=')[1])
        if l.startswith('AUBIO_PATCH_VERSION'):
            AUBIO_PATCH_VERSION = int(l.split('=')[1])
        if l.startswith('AUBIO_VERSION_STATUS'):
            AUBIO_VERSION_STATUS = l.split('=')[1].strip()[1:-1]

    if AUBIO_MAJOR_VERSION is None or AUBIO_MINOR_VERSION is None \
            or AUBIO_PATCH_VERSION is None:
        raise SystemError("Failed parsing VERSION file.")

    verstr = '.'.join(map(str, [AUBIO_MAJOR_VERSION,
                                     AUBIO_MINOR_VERSION,
                                     AUBIO_PATCH_VERSION]))

    
    # append sha to version in alpha release
    # MAJ.MIN.PATCH.{~git<sha> , ''}
    if '~alpha' in AUBIO_VERSION_STATUS :
        AUBIO_GIT_SHA = get_git_revision_hash()
        if AUBIO_GIT_SHA:
            AUBIO_VERSION_STATUS = '~git'+AUBIO_GIT_SHA

    if AUBIO_VERSION_STATUS is not None :
        verstr += AUBIO_VERSION_STATUS
    return verstr

def get_aubio_pyversion():
    # convert to version for python according to pep 440
    # see https://www.python.org/dev/peps/pep-0440/
    verstr = get_aubio_version()
    spl = verstr.split('~')
    if len(spl)==2:
        verstr = spl[0] + '+a1.'+spl[1]

    # TODO: add rc, .dev, and .post suffixes, add numbering
    return verstr



def get_git_revision_hash( short=True):
    def which(program):
    
        def is_exe(fpath):
            return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

        fpath, fname = os.path.split(program)
        if fpath:
            if is_exe(program):
                return program
        else:
            for path in os.environ["PATH"].split(os.pathsep):
                path = path.strip('"')
                exe_file = os.path.join(path, program)
                if is_exe(exe_file):
                    return exe_file

        return None

    if not which('git') :
        # print('no git found on this system : can\'t get sha')
        return ""
    if not os.path.isdir('.git'):
        # print('Version : not in git repository : can\'t get sha')
        return ""

    import subprocess
    aubio_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(aubio_dir):
        raise SystemError("git / root folder not found")
    gitcmd = ['git','-C',aubio_dir ,'rev-parse']
    if short:
      gitcmd.append('--short')
    gitcmd.append('HEAD')
    outCmd = subprocess.check_output(gitcmd).strip().decode('utf8')
    return outCmd



# append sha to version in alpha release
if AUBIO_VERSION_STATUS and '~alpha' in AUBIO_VERSION_STATUS :
    AUBIO_GIT_SHA = get_git_revision_hash()
    if AUBIO_GIT_SHA:
        AUBIO_VERSION_STATUS = '~git'+AUBIO_GIT_SHA