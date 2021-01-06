from setuptools import find_packages, setup

# with open("README.md", "r") as fh:
#     long_description = fh.read()

setup(
    name="smg-relocalisation",
    version="0.0.1",
    author="Stuart Golodetz",
    author_email="stuart.golodetz@cs.ox.ac.uk",
    description="Various approaches to camera relocalisation",
    long_description="",  #long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sgolodetz/smg-relocalisation",
    packages=find_packages(include=["smg.relocalisation", "smg.relocalisation.*"]),
    include_package_data=True,
    install_requires=[
        "numpy",
        "opencv-contrib-python==3.4.2.16",
        "PyOpenGL",
        "vg"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
