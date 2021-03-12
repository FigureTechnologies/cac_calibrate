from setuptools import setup, find_packages

setup(
    name="cac_calibrate",
    version="0.01",
    description="",
    author="Andrew Hah",
    author_email="hahdawg@yahoo.com",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "pandas>=1.2.2",
        "numpy>=1.19.2",
        "statsmodels>=0.12.2",
        "scikit_learn>=0.24.1"
    ],
    zip_safe=False
)
