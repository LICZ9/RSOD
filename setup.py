#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import platform
import shutil
import sys
import warnings
from setuptools import find_packages, setup

import torch
from torch.utils.cpp_extension import (BuildExtension, CppExtension,
                                       CUDAExtension)


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


version_file = 'mmdet/version.py'


def get_version():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


def make_cuda_ext(name, module, sources, sources_cuda=[]):

    define_macros = []
    extra_compile_args = {'cxx': []}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        sources += sources_cuda
    else:
        print(f'Compiling {name} without CUDA')
        extension = CppExtension

    return extension(
        name=f'{module}.{name}',
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)


def parse_requirements(fname='requirements.txt', with_version=True):
    """Parse the package dependencies listed in a requirements file but strips
    specific versioning information.

    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if True include version specs

    Returns:
        List[str]: list of requirements items

    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    import re
    import sys
    from os.path import exists
    require_fpath = fname

    def parse_line(line):
        """Parse information from a line in a requirements text file."""
        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = line.split(' ')[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            elif '@git+' in line:
                info['package'] = line
            else:
                # Remove versioning from the package
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip,
                                                     rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest  # NOQA
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info['package']]
                if with_version and 'version' in info:
                    parts.extend(info['version'])
                if not sys.version.startswith('3.4'):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get('platform_deps')
                    if platform_deps is not None:
                        parts.append(';' + platform_deps)
                item = ''.join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages


def add_mim_extension():
    """Add missing MIM resources without replacing existing destinations.

    Editable installs link new entries on platforms supporting symlinks.
    Existing files and links take precedence over the source tree.
    """

    # parse installment mode
    if 'develop' in sys.argv or 'editable_wheel' in sys.argv:
        # installed by `pip install -e .`
        if platform.system() == 'Windows':
            # set `copy` mode here since symlink fails on Windows.
            mode = 'copy'
        else:
            mode = 'symlink'
    elif 'sdist' in sys.argv or 'bdist_wheel' in sys.argv:
        # installed by `pip install .`
        # or create source distribution by `python setup.py sdist`
        mode = 'copy'
    else:
        return

    filenames = [
        'tools', 'configs', 'demo', 'model-index.yml', 'dataset-index.yml'
    ]
    repo_path = osp.abspath(osp.dirname(__file__))
    mim_path = osp.join(repo_path, 'mmdet', '.mim')
    if osp.islink(mim_path):
        warnings.warn('Preserving the existing MIM resource symbolic link.')
        return
    os.makedirs(mim_path, exist_ok=True)

    def add_missing(src_path, tar_path):
        if (osp.basename(src_path) == '__pycache__'
                or src_path.endswith(('.pyc', '.pyo'))):
            return
        if osp.islink(src_path):
            warnings.warn(f'Skipping symbolic-link resource source: {src_path}')
            return
        if osp.lexists(tar_path):
            if (osp.islink(tar_path)
                    or not (osp.isdir(src_path) and osp.isdir(tar_path))):
                return
        elif mode == 'symlink':
            try:
                src_relpath = osp.relpath(src_path, osp.dirname(tar_path))
                os.symlink(src_relpath, tar_path)
            except FileExistsError:
                pass
            return
        elif osp.isdir(src_path):
            try:
                os.mkdir(tar_path)
            except FileExistsError:
                if osp.islink(tar_path) or not osp.isdir(tar_path):
                    return
        elif osp.isfile(src_path):
            try:
                with open(src_path, 'rb') as source:
                    with open(tar_path, 'xb') as destination:
                        shutil.copyfileobj(source, destination)
            except FileExistsError:
                return
            shutil.copystat(src_path, tar_path)
            return
        else:
            warnings.warn(f'Cannot copy file {src_path}.')
            return

        for name in sorted(os.listdir(src_path)):
            add_missing(osp.join(src_path, name), osp.join(tar_path, name))

    for filename in filenames:
        src_path = osp.join(repo_path, filename)
        if osp.exists(src_path):
            add_missing(src_path, osp.join(mim_path, filename))


if __name__ == '__main__':
    add_mim_extension()
    setup(
        name='mmdet',
        version=get_version(),
        description='OpenMMLab Detection Toolbox and Benchmark',
        long_description=readme(),
        long_description_content_type='text/markdown',
        author='MMDetection Contributors',
        author_email='openmmlab@gmail.com',
        keywords='computer vision, object detection',
        url='https://github.com/open-mmlab/mmdetection',
        packages=find_packages(exclude=('configs', 'tools', 'demo')),
        include_package_data=True,
        python_requires='>=3.9',
        classifiers=[
            'Development Status :: 5 - Production/Stable',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.9',
        ],
        license='Apache License 2.0',
        install_requires=parse_requirements('requirements/runtime.txt'),
        extras_require={
            'all': parse_requirements('requirements.txt'),
            'tests': parse_requirements('requirements/tests.txt'),
            'build': parse_requirements('requirements/build.txt'),
            'optional': parse_requirements('requirements/optional.txt'),
            'mim': parse_requirements('requirements/mminstall.txt'),
            'tracking': parse_requirements('requirements/tracking.txt'),
            'multimodal': parse_requirements('requirements/multimodal.txt'),
        },
        ext_modules=[],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False)
