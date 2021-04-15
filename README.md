# Skin Lesion Classifier
A Skin Lesion Analysis Towards Melanoma Detection

![](https://img.shields.io/badge/Academical%20Project-Yes-success)
![](https://img.shields.io/badge/License-Free%20To%20Use-green)
![](https://img.shields.io/badge/Made%20with-Python-blue)
![](https://img.shields.io/badge/ISIC%20Challenge-2017-lightgrey)
![](https://img.shields.io/badge/Maintained-No-red)

## Description 

This project aims to propose a form dealing with the classification of skin lesions into 3 unique diagnoses (melanoma, nevus, and seborrheic keratosis) based on deep neural networks working
together.
The purpose of the system developed was based on the [ISIC Challenge of 2017](https://challenge.isic-archive.com/) and uses the skin lesions datasets available online.

The validation score for the Melanoma Classifier was 0.80 and for the Seborrheic Keratosis Classifier was 0.75.
The system developed achieved a lower average accuracy score due to the combination process of the two classifiers.

## Repository Structure 

/normalized-dataset - contains parts of the processed dataset used

/paper - contains the written paper on the conducted analysis

/src - contains the source code written in Python

## Additional Resources

<img src="https://github.com/FilipePires98/SkinLesionClassifier/blob/master/paper/SkinLesions.jpg" width="360px">
Examples of input images with contrasting features, paired with the same images after the preprocessing phase.

![Architecture](https://github.com/FilipePires98/SkinLesionClassifier/blob/master/paper/ModelArchitecture_1Classifier.jpg)

Architecture of each classifier for the proposed classification model.

## Authors

The authors of this repository are Filipe Pires and João Alegria, and the project was developed for the Machine Learning Course of the licenciate's degree in Informatics Engineering of the University of Aveiro.

For further information, please read our [paper](https://github.com/FilipePires98/SkinLesionClassifier/blob/master/paper/Paper.pdf) or contact us at filipesnetopires@ua.pt or joao.p@ua.pt.




