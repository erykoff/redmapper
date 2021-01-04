# -*- coding: utf-8 -*-

from setuptools import setup, find_packages, Extension, Command
import numpy,os,glob

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

exec(open('redmapper/_version.py').read())

scripts = ['bin/redmapper_run_zred_pixel.py',
           'bin/redmapper_run_redmapper_pixel.py',
           'bin/redmapper_batch.py',
           'bin/redmapper_make_zred_bkg.py',
           'bin/redmapper_calibrate.py',
           'bin/redmapper_consolidate_run.py',
           'bin/redmagic_calibrate.py',
           'bin/redmagic_run.py',
           'bin/redmapper_convert_mask_to_healsparse.py',
           'bin/redmapper_convert_depthfile_to_healsparse.py',
           'bin/redmapper_run_many_pixels_same_node.py',
           'bin/redmapper_build_docker.py',
           'bin/redmapper_consolidate_runcat.py',
           'bin/redmapper_runcat_pixel.py',
           'bin/redmapper_run_zmask_pixel.py',
           'bin/redmapper_run_zscan_pixel.py',
           'bin/redmapper_generate_randoms.py',
           'bin/redmapper_weight_randoms.py',
           'bin/redmapper_predict_memory.py']

include_dirs = [numpy.get_include()]

ext_modules=[]

# solver_nfw
solver_nfw_sources=['redmapper/solver_nfw/solver_nfw_pywrap.c',
                    'redmapper/solver_nfw/solver_nfw.c',
                    'redmapper/solver_nfw/nfw_weights.c']
solver_nfw_module = Extension('redmapper.solver_nfw._solver_nfw_pywrap',
                              extra_compile_args=['-std=gnu99'],
                              sources=solver_nfw_sources,
                              include_dirs=include_dirs)
ext_modules.append(solver_nfw_module)

# chisq_dist
chisq_dist_sources=['redmapper/chisq_dist/chisq_dist.c',
                    'redmapper/chisq_dist/chisq_dist_pywrap.c']
chisq_dist_module = Extension('redmapper.chisq_dist._chisq_dist_pywrap',
                              extra_compile_args=['-std=gnu99',os.path.expandvars('-I${GSLI}')],
                              extra_link_args=[os.path.expandvars('-L${GSLL}')],
                              libraries=['gsl', 'gslcblas'],
                              sources=chisq_dist_sources,
                              include_dirs=include_dirs)
ext_modules.append(chisq_dist_module)

class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info')

setup(
    name='redmapper',
    version=__version__,
    description='Public, Python implementation of redMaPPer',
    long_description=readme,
    author='Eli Rykoff, Brett Harvey',
    author_email='erykoff@slac.stanford.edu, rbharvey@stanford.edu',
    url='https://github.com/erykoff/redmapper',
    license=license,
    ext_modules=ext_modules,
    scripts=scripts,
    install_requires=['numpy'],
    packages=find_packages(exclude=('tests', 'docs')),
    include_package_data=True,
    cmdclass={'clean': CleanCommand}
)

