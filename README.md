# CV Anomaly Detection

This is a repo for Computer Vision anomaly detection studies.

## Datasets
MVTec AD datasets : Download from [MVTec website](https://www.mvtec.com/company/research/datasets/mvtec-ad/)

>Paul Bergmann, Michael Fauser, David Sattlegger, and Carsten Steger,
>"A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection",
>IEEE Conference on Computer Vision and Pattern Recognition, 2019

# Poetry

To install Poetry, follow the steps on [the official site](https://python-poetry.org/docs/#installing-with-the-official-installer)

## Using Poetry

### Check current version
```shell
poetry --version
```

### Installing dependencies
```shell
poetry install
```

### [Activating the virtual environment](https://python-poetry.org/docs/basic-usage/#activating-the-virtual-environment)
```shell
poetry shell
```

### adding dependencies
```shell
poetry add opencv-python
```

### updating dependencies
```shell
poetry update
```
### Generating requirements
```shell
poetry export --without-hashes --format=requirements.txt > requirements.txt
```
