import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="segpack",
    version="1.0.0",
    author="Jaemin Son",
    author_email="7109417@hyundai.com",
    description="Segmentation package for forward validation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.hmckmc.co.kr/airlab/segpack",
    project_urls={
        "Bug Tracker": "github.hmckmc.co.kr/airlab/segpack/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
)
