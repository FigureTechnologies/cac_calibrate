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
        "pandas",
        "numpy",
        "statsmodels",
        "scikit_learn"
    ],
    zip_safe=False
)
