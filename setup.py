from setuptools import setup

setup(
    name='InLIFEsim',
    version='0.0.1',
    description='Instrumental noise simulator software for the Large Interferometer For Exoplanets (LIFE)',
    author='Felix Dannert',
    author_email='fdannert@ethz.ch',
    url='https://github.com/fdannert/InLIFEsim',
    packages=['inlifesim'],
    include_package_data=True,
    install_requires=['numpy',
                      'scipy',
                      'pandas',
                      'tqdm',
                      'xarray'
                      ],
    license='GPLv3',
    zip_safe=False,
    keywords='inlifesim',
    python_requires='~=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.8',
    ]
)