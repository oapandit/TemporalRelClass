# Temporal Relation Classification
This code implements all the experiments which are described in our paper "Learning Rich Event Representations and Interactions for Temporal Relation Classification", appeared in ESANN 2019.

# Steps to Replicate experiments
1. The current code is compiled on python 3.8, so use this version.
2. Use yml file to setup all the libraries.
3. Download the TimeML annotated data.
4. Download Google word2vec and facebook fasttext embeddings.
5. Change the base directory and appropriate word embeddings directory path.
6. Execute the tre_system.py file. The main experiemnt is present with train_neural_model, run it to see the results from the paper.
7. For other experiments from the paper use corresponding methods from the same file. 

