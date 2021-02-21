# Portfolio
Portfolio of data science projects that include:

1) <a href ="https://nbviewer.jupyter.org/github/pnanimal/Springboard/blob/master/FatalEncounters.ipynb">Fatal Encounters</a>: An analysis of people killed during an interaction with a US police officer, both on and off-duty. EDA on males and transgender individuals showed that most died by gunshot, while females died by vehicle. The number of police-related fatalities over 20 years increased by approximately 111%, far exceeding population growth. This increase nationwide was largely composed of gunshot-related deaths and not driven by the agencies that were responsible for the highest number of deaths, but by others. The number of police-related fatalities do not seem to have been impacted by the media attention and protests connected with recent deaths which include that of George Floyd. Logistic regression is used to provide weights with which to identify the most important features for modeling deaths related to gunshot. These were improved after reducing the data to a state-specific analysis, which provided insights for TX, CA, GA, NY, AZ and FL, highlighting features with both positive and negative weights, to indicate their contributions to gunshot-related outcomes. In Texas, the highest weights were associated with border activities, Houston and the area around Fort Worth. In CA, the second highest weight corresponds to the small city of Redding, which has been noted in the media for its large number of police-related fatalities. In AZ, the most negative weight corresponded to the race Asian/Pacific Islander, and Navajo County and the city of Holbrook. In FL, the highest positive weights corresponded to the gender male, communities on the beach, and the year. Insights from these models can be used to focus efforts and increase community understanding of the underlying problems that could lead to fatal encounters and how to start reducing them over time. In particular, modeling can be used to focus on discrepancies between similar areas in population or location that have different feature weights, to understand regional behaviors and how they affect national trends, and to plan and budget resources towards reducing discrepancies where appropriate.

2) <a href="https://github.com/pnanimal/Springboard/blob/master/AnimalShelter.ipynb">Austin Animal Shelter Outcomes</a>: The animal shelter in Austin, TX is the largest no-kill animal shelter in the US. This project analyzes the animals that were housed there and what happened to them from October 2013 through February 1, 2018 via a dataset released through Kaggle. This project showcases exploratory data analysis, data wrangling, visualization and story telling to answer questions including: What kinds of animals are at the shelter? What happened to them when they got there? What kind of animals are euthanized and why? 

3) <a href="https://github.com/pnanimal/Portfolio/blob/main/AudioDeepLearning/README.md">Deep Learning for Audio Classification</a>: Analysis of three different deep learning algorithms on a Kaggle dataset containing audio recordings of birds and frogs from 24 different species. These audio files are processed to extract audio features including spectrograms, spectral bandwidth, spectral centroids, Mel-frequency cepstrum (MFCC’s), and chroma temperature. Species which had less than 50 samples are augmented to 50 samples using a random choice between adding background noise, repeating the signal or adding additional recording time to the sample so as not to disadvantage rarer species.These features are appropriately shaped and/or selected to test the performance of three deep learning models: 1) an artificial neural net (ANN), 2) a convolutional neural net (CNN) with and without transfer learning and/or augmentation, and 3) a recurrent neural net (RNN). Overall, the ANN had the second best performance with 91.5% accuracy on the training, and 93.4% accuracy on the test set using a subset of the features. The CNN without transfer learning (augmentation didn’t change the final performance) had the best performance with an accuracy of 96.4% on the training, and 96.7% on the test set. The RNN using only the MFCC’s achieved an accuracy of 94.6% on the training, and 93.0% on the test set.
