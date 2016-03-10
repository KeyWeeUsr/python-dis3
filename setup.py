from setuptools import setup


HOMEPAGE = 'https://gitlab.com/tomxtobin/python-dis3'


setup(
    author='Tom X. Tobin',
    author_email='tomxtobin@tomxtobin.com',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'License :: OSI Approved :: Python Software Foundation License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 2 :: Only',
        'Topic :: Software Development :: Disassemblers',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    description="""Python 2.7 backport of the "dis" module from Python 3.5+""",
    license='MIT',
    long_description='See {} for details.'.format(HOMEPAGE),
    name='dis3',
    py_modules=['dis3'],
    url=HOMEPAGE,
    version='0.1.1',
)
