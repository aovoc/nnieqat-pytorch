import os
import shutil
import subprocess
import sys
import tempfile
from distutils import ccompiler


def print_warning(*lines):
    print('**************************************************')
    for line in lines:
        print('*** WARNING: %s' % line)
    print('**************************************************')


def get_path(key):
    return os.environ.get(key, '').split(os.pathsep)


def search_on_path(filenames):
    for p in get_path('PATH'):
        for filename in filenames:
            full = os.path.join(p, filename)
            if os.path.exists(full):
                return os.path.abspath(full)
    return None


minimum_cuda_version = 10010
maxinum_cuda_version = 10030
minimum_cudnn_version = 7000


def get_compiler_setting():
    nvcc_path = search_on_path(('nvcc', 'nvcc.exe'))
    cuda_path_default = None
    if nvcc_path is None:
        print_warning('nvcc not in path.', 'Please set path to nvcc.')
    else:
        cuda_path_default = os.path.normpath(
            os.path.join(os.path.dirname(nvcc_path), '..'))

    cuda_path = os.environ.get('CUDA_PATH', '')  # Nvidia default on Windows
    if len(cuda_path) > 0 and cuda_path != cuda_path_default:
        print_warning('nvcc path != CUDA_PATH',
                      'nvcc path: %s' % cuda_path_default,
                      'CUDA_PATH: %s' % cuda_path)

    if not os.path.exists(cuda_path):
        cuda_path = cuda_path_default

    if not cuda_path and os.path.exists('/usr/local/cuda'):
        cuda_path = '/usr/local/cuda'

    include_dirs = []
    library_dirs = []
    define_macros = []

    if cuda_path:
        include_dirs.append(os.path.join(cuda_path, 'include'))
        if sys.platform == 'win32':
            library_dirs.append(os.path.join(cuda_path, 'bin'))
            library_dirs.append(os.path.join(cuda_path, 'lib', 'x64'))
        else:
            library_dirs.append(os.path.join(cuda_path, 'lib64'))
            library_dirs.append(os.path.join(cuda_path, 'lib'))
    if sys.platform == 'darwin':
        library_dirs.append('/usr/local/cuda/lib')

    return {
        'include_dirs': include_dirs,
        'library_dirs': library_dirs,
        'define_macros': define_macros,
        'language': 'c++',
    }


def check_cuda_version():
    compiler = ccompiler.new_compiler()
    settings = get_compiler_setting()
    try:
        out = build_and_run(compiler,
                            '''
        #include <cuda.h>
        #include <stdio.h>
        int main(int argc, char* argv[]) {
          printf("%d", CUDA_VERSION);
          return 0;
        }
        ''',
                            include_dirs=settings['include_dirs'])

    except Exception as e:
        print_warning('Cannot check CUDA version', str(e))
        return False

    cuda_version = int(out)
    if cuda_version < minimum_cuda_version:
        print_warning('CUDA version is too old: %d' % cuda_version,
                      'CUDA v10.1 or CUDA v10.2 is required')
        return False
    if cuda_version > maxinum_cuda_version:
        print_warning('CUDA version is too new: %d' % cuda_version,
                      'CUDA v10.1 or CUDA v10.2 is required')

    return True


def check_cudnn_version():
    compiler = ccompiler.new_compiler()
    settings = get_compiler_setting()
    try:
        out = build_and_run(compiler,
                            '''
        #include <cudnn.h>
        #include <stdio.h>
        int main(int argc, char* argv[]) {
          printf("%d", CUDNN_VERSION);
          return 0;
        }
        ''',
                            include_dirs=settings['include_dirs'])

    except Exception as e:
        print_warning('Cannot check cuDNN version\n{0}'.format(e))
        return False

    cudnn_version = int(out)
    if cudnn_version < minimum_cudnn_version:
        print_warning('cuDNN version is too old: %d' % cudnn_version,
                      'cuDNN v7 or newer is required')
        return False

    return True


def build_and_run(compiler,
                  source,
                  libraries=(),
                  include_dirs=(),
                  library_dirs=()):
    temp_dir = tempfile.mkdtemp()

    try:
        fname = os.path.join(temp_dir, 'a.cpp')
        with open(fname, 'w') as f:
            f.write(source)

        objects = compiler.compile([fname],
                                   output_dir=temp_dir,
                                   include_dirs=include_dirs)

        try:
            postargs = ['/MANIFEST'] if sys.platform == 'win32' else []
            compiler.link_executable(objects,
                                     os.path.join(temp_dir, 'a'),
                                     libraries=libraries,
                                     library_dirs=library_dirs,
                                     extra_postargs=postargs,
                                     target_lang='c++')
        except Exception as e:
            msg = 'Cannot build a stub file.\nOriginal error: {0}'.format(e)
            raise Exception(msg)

        try:
            out = subprocess.check_output(os.path.join(temp_dir, 'a'))
            return out

        except Exception as e:
            msg = 'Cannot execute a stub file.\nOriginal error: {0}'.format(e)
            raise Exception(msg)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
