# Sentiment Analysis of Movie Reviews using Word2Vec & LSTM Network (in PyTorch)

This is my implementation of Sentiment Analysis using Long-Short Term Memory (LSTM) Network. The code performs:

1. Loading and pre-processing raw reviews & labels data
2. Building a deep neural network including Word2Vec embeddings and LSTM layers
3. Test the performance of the model in classifying a random review as postive or negative.
   
   
   
Main components of the network:
I. **Word2Vec Embedding** - used to reduce dimensionality, as there are tens of thousands of words in the entire vocabulary of all reviews. Each of those words are represented as vectors in 400-dimension space.
II. **LSTM Layers** - used to look at the review texts as the sequence of inputs, rather than individual, in order to take advantage of the bigger context of the text.



## Repository 

This repository contains:
* **sentiment_analysis_LSTM.py** : Complete code for implementing the sentiment analysis of movie reviews using LSTM network
* **data folder** : includes reviews.txt (contains all reivews) & labels.txt (contains all corresponding labels)
		
		
			
## List of Hyperparameters used:

* Batch Size = **50**
* Sequence Length for Movie Reviews = **200**  
* Embedding Dimension = **400**
* Number of hidden nodes in LSTM = **256**
* Number of LSTM Layers = **2**
* Learning Rate = **0.001**
* Gradient Clip Maximum Threshold= **5**
* Number of Epochs = **4**



## Sources

I referenced the following sources for building & debugging the final model :

* https://github.com/udacity/deep-learning-v2-pytorch



