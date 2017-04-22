from setuptools import setup

setup(
    name='rstools',
    version='0.1',
    description='Research tools.',
    url='https://github.com/Scitator/rstools',
    author='Kolesnikov Sergey',
    author_email='scitator@gmail.com',
    license='GPL3',
    packages=['rstools'],
    package_dir={'rstools': './rstools'},
    install_requires=[
        'tensorflow==1.0.0',
    ],
    zip_safe=False)
