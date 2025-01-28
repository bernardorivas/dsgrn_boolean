from setuptools import setup, find_packages

setup(
    name='dsgrn_boolean',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'dsgrn',
        'pychomp2',
        'numpy',
        'jax',
        'jaxlib',
        'jupyter',
    ],
    python_requires='>=3.7',
    author='Bernardo Rivas',
    author_email='bernardo.dopradorivas@utoledo.edu',
    description='DSGRN Boolean Analysis Tools',
    # long_description=open('README.md').read(),
    # long_description_content_type='text/markdown',
    # url='https://github.com/bernardorivas/dsgrn_boolean', 
)