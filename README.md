# Human Activity Recognition (HAR) Project

## Overview
This project involves Human Activity Recognition (HAR) using accelerometer data. The goal is to classify human activities based on data collected from smartphones. The project includes data preprocessing, exploratory data analysis, machine learning models, and prompt engineering for large language models.

## Project Structure
- **Data Preprocessing**: Scripts to combine and create datasets from the raw accelerometer data.
- **Exploratory Data Analysis (EDA)**: Visualizations and analysis to understand the data and its features.
- **Decision Trees**: Implementation and evaluation of decision tree classifiers using raw and feature-engineered data.
- **Prompt Engineering**: Utilizing zero-shot and few-shot learning for activity classification.
- **Data Collection in the Wild**: Collecting and analyzing personal data to evaluate model performance.

## Current Status

### Completed
- **Data Preprocessing**: The scripts `CombineScript.py` and `MakeDataset.py` have been successfully executed to organize and split the dataset into training, testing, and validation sets.
- **Exploratory Data Analysis**: Initial EDA has been performed. Specific steps completed include [brief description of the completed EDA step].

### To Do
- **Continue EDA**:
  - Plot waveforms for different activity classes.
  - Perform PCA on various feature sets.
  - Calculate and analyze the correlation matrix.

- **Decision Trees**:
  - Train decision tree models using different feature sets.
  - Evaluate and compare model performance.
  - Analyze performance with varying tree depths.

- **Prompt Engineering**:
  - Implement zero-shot and few-shot learning methods.
  - Compare performance with decision tree models.

- **Data Collection in the Wild**:
  - Collect and preprocess personal accelerometer data.
  - Evaluate model performance on personal data.

## Scripts and Files
- `CombineScript.py`: Organizes accelerometer data into a combined dataset.
- `MakeDataset.py`: Creates training, testing, and validation datasets.
- `EDA.ipynb`: Jupyter Notebook for exploratory data analysis.
- `decision_tree.py`: Implementation of decision tree models.
- `prompt_engineering.py`: Scripts for zero-shot and few-shot learning.
- `data_collection.py`: Scripts for data collection and evaluation.

## How to Run
1. Ensure you have the necessary dependencies installed (see `requirements.txt`).
2. Place `CombineScript.py` and `MakeDataset.py` in the same folder as the UCI dataset.
3. Run the scripts to preprocess the data:
    ```bash
    python CombineScript.py
    python MakeDataset.py
    ```
4. For EDA, open `EDA.ipynb` in a Jupyter Notebook and run the cells.
5. Follow the instructions in `decision_tree.py` and `prompt_engineering.py` to train and evaluate models.

## Results
- **EDA**: [Link to or description of visualizations and findings]
- **Decision Trees**: [Link to or description of model performance metrics]
- **Prompt Engineering**: [Link to or description of results]

## Notes
- Ensure to keep the API key secure. Do not share or upload it to any public repository.
- The project will be updated as more tasks are completed.

## Contact
For any questions or suggestions, please contact Soham Gaonkar at sohamgaonkar2005@gmail.com.
