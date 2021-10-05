from setuptools import setup, find_packages

setup(
    name="cac_calibrate",
    version="0.02",
    description="",
    author="Andrew Hah",
    author_email="hahdawg@yahoo.com",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "pandas",
        "pytest",
        "numpy",
        "ray==1.4.1",
        "statsmodels",
        "scikit_learn"
    ],
    zip_safe=False
)
