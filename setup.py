from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="ofco",
    version="0.1",
    packages=["ofco"],
    entry_points={"console_scripts": ["ofco = ofco.main:cli"]},
    author="Florian Aymanns",
    author_email="florian.ayamnns@epfl.ch",
    description="Optic flow based motion correction for Calcium two-photon images.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NeLy-EPFL/ofco",
    install_requires=[
        "numpy",
        "scipy",
        "scikit-image",
        "opencv-python",
        "docopt",
        "sphinx",
    ],
)
