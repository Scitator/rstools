from setuptools import setup

setup(
    name='rstools',
    version='0.1',
    description='Research tools.',
    url='https://github.com/Scitator/rstools',
    author='Kolesnikov Sergey',
    author_email='scitator@gmail.com',
    license='GPL3',
    packages=['rstools', 'rstools.tf', 'rstools.utils', 'rstools.visualization'],
    package_dir={
        'rstools': './rstools',
        'rstools.tf': './rstools/tf',
        'rstools.utils': './rstools/utils',
        'rstools.visualization': './rstools/visualization',
    },
    install_requires=[
        'tensorflow==1.0.0',
    ],
    zip_safe=False)
