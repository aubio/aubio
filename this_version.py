
import os


__version_info = {}


def get_version_info():
    # read from VERSION
    # return dictionary filled with content of version

    if not __version_info:
        this_file_dir = os.path.dirname(os.path.abspath(__file__))
        version_file = os.path.join(this_file_dir, 'VERSION')

        if not os.path.isfile(version_file):
            raise SystemError("VERSION file not found.")

        for l in open(version_file).readlines():

            if l.startswith('AUBIO_MAJOR_VERSION'):
                __version_info['AUBIO_MAJOR_VERSION'] = int(l.split('=')[1])
            if l.startswith('AUBIO_MINOR_VERSION'):
                __version_info['AUBIO_MINOR_VERSION'] = int(l.split('=')[1])
            if l.startswith('AUBIO_PATCH_VERSION'):
                __version_info['AUBIO_PATCH_VERSION'] = int(l.split('=')[1])
            if l.startswith('AUBIO_VERSION_STATUS'):
                __version_info['AUBIO_VERSION_STATUS'] = \
                    l.split('=')[1].strip()[1:-1]

            if l.startswith('LIBAUBIO_LT_CUR'):
                __version_info['LIBAUBIO_LT_CUR'] = int(l.split('=')[1])
            if l.startswith('LIBAUBIO_LT_REV'):
                __version_info['LIBAUBIO_LT_REV'] = int(l.split('=')[1])
            if l.startswith('LIBAUBIO_LT_AGE'):
                __version_info['LIBAUBIO_LT_AGE'] = int(l.split('=')[1])

        if len(__version_info) < 6:
            raise SystemError("Failed parsing VERSION file.")

        # switch version status with commit sha in alpha releases
        if __version_info['AUBIO_VERSION_STATUS'] and \
                '~alpha' in __version_info['AUBIO_VERSION_STATUS']:
            AUBIO_GIT_SHA = get_git_revision_hash()
            if AUBIO_GIT_SHA:
                __version_info['AUBIO_VERSION_STATUS'] = '~git+' + AUBIO_GIT_SHA

    return __version_info


def get_aubio_version_tuple():
    d = get_version_info()
    return (d['AUBIO_MAJOR_VERSION'],
            d['AUBIO_MINOR_VERSION'],
            d['AUBIO_PATCH_VERSION'])


def get_libaubio_version_tuple():
    d = get_version_info()
    return (d['LIBAUBIO_LT_CUR'], d['LIBAUBIO_LT_REV'], d['LIBAUBIO_LT_AGE'])


def get_libaubio_version():
    return '%s.%s.%s' % get_libaubio_version_tuple()


def get_aubio_version(add_status=True):
    # return string formatted as MAJ.MIN.PATCH{~git<sha> , ''}
    vdict = get_version_info()
    verstr = '%s.%s.%s' % get_aubio_version_tuple()
    if add_status and vdict['AUBIO_VERSION_STATUS']:
        verstr += vdict['AUBIO_VERSION_STATUS']
    return str(verstr)


def get_aubio_pyversion(add_status=True):
    # convert to version for python according to pep 440
    # see https://www.python.org/dev/peps/pep-0440/
    # outputs MAJ.MIN.PATCH+a0{.git<sha> , ''}
    vdict = get_version_info()
    verstr = '%s.%s.%s' % get_aubio_version_tuple()
    if add_status and vdict['AUBIO_VERSION_STATUS']:
        if '~git' in vdict['AUBIO_VERSION_STATUS']:
            verstr += "+a0." + vdict['AUBIO_VERSION_STATUS'][1:]
        elif '~alpha' in vdict['AUBIO_VERSION_STATUS']:
            verstr += "+a0"
        else:
            raise SystemError("Aubio version statut not supported : %s" %
                              vdict['AUBIO_VERSION_STATUS'])
    return verstr


def get_git_revision_hash(short=True):

    if not os.path.isdir('.git'):
        # print('Version : not in git repository : can\'t get sha')
        return None

    import subprocess
    aubio_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(aubio_dir):
        raise SystemError("git / root folder not found")
    gitcmd = ['git', '-C', aubio_dir, 'rev-parse']
    if short:
        gitcmd.append('--short')
    gitcmd.append('HEAD')
    try:
        outCmd = subprocess.check_output(gitcmd).strip().decode('utf8')
    except Exception as e:
        print('git command error :%s' % e)
        return None
    return outCmd
