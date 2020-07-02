# -*- coding: utf-8 -*-


from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='jpmsklp',
    version='0.1.0',
    description="Jeremy P Mann's custom pipe(lines)",
    long_description=readme,
    author='Jeremy P Mann',
    author_email='jmann277@gmail.com',
    url='https://github.com/jmann277/jpm-skl-pipelines',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

