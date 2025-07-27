from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="retico-gemini",
    version="0.1.0",
    author="Émile Alexandre",
    author_email="emile.alexandre9@etu.univ-lorraine.fr",
    description="A Retico module that communicates with the Gemini API over text and audio input.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mi-1000/retico-gemini",
    packages=find_packages(),
    install_requires=requirements,
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7,<=3.12",
)
