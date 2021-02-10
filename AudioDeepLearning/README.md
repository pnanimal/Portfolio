
<p>
Springboard -- DSC
</p>
<h1><strong><b>Audio Classification through Deep Learning</b></strong>
<p>Analysis and Modeling </h1>
<p><h2>Papia Nandi
</p>
<p>February 2021
</p>
<hr align="center" color="orange" size="3" width="500">
<p>
<div align="center">
  <img src="https://github.com/pnanimal/Portfolio/blob/images/AudioDeepLearning/Titmouse.jpg"><br><br>
</div>
<h1>Introduction</h1>

* Audio classification through deep learning analyzes three different deep learning algorithms on a [Kaggle](https://www.kaggle.com/c/rfcx-species-audio-detection) dataset containing audio recordings of birds and frogs from 24 different species.
* These audio files are processed to extract audio features including <b>spectrograms, spectral bandwidth, spectral centroids, Mel-frequency cepstrum (MFCC’s)</b>, and <b>chroma temperature</b>.  
* Species which had less than 50 samples are <b>augmented to 50 samples </b>using a random choice between adding background noise, repeating the signal or adding additional recording time to the sample so as not to disadvantage rarer species.
* These features are appropriately shaped and/or selected to test the performance of three deep learning models: 1) an artificial neural net (<b>ANN</b>), 2) a convolutional neural net (<b>CNN</b>) with and without transfer learning and/or augmentation, and 3) a recurrent neural net (<b>RNN</b>). 
</p>
<p><h3>Jupyter Notebooks contain reproducible code used to generate and test the different deep learning models are available at the following links: </h3>

  * [Exploratory Data Analysis](https://github.com/pnanimal/Portfolio/blob/main/AudioDeepLearning/birdfrog-eda.ipynb)
  * [ANN](https://github.com/pnanimal/Portfolio/blob/main/AudioDeepLearning/birdfrogann.ipynb)
  * [CNN](https://github.com/pnanimal/Portfolio/blob/main/AudioDeepLearning/birdfrogcnn.ipynb)
  * [RNN](https://github.com/pnanimal/Portfolio/blob/main/AudioDeepLearning/birdfrogrnn.ipynb)

Additional Jupyter Notebooks were used to investigate incorporating [transfer learning](https://github.com/pnanimal/Portfolio/blob/main/AudioDeepLearning/birdfrogcnnwtl.ipynb) from the University of Oxford’s VGG16 solution to the ImageNet challenge into the final CNN model, and running the CNN model [without augmentation](https://github.com/pnanimal/Portfolio/blob/main/AudioDeepLearning/birdfrogcnnnoaug.ipynb). These results were not used in the final models.

<P>Overall, the ANN had the second best performance with <b>91.5% accuracy</b> on the training, and <b>93.4%</b> accuracy on the test set using a subset of the features. The CNN without transfer learning (augmentation didn’t change the final performance) had the best performance with an accuracy of <b?96.4%</b> on the training, and <b>96.7%</b> on the test set. The RNN using only the MFCC’s achieved an accuracy of <b>94.6%</b> on the training, and <b>93.0%</b> on the test set.
</P>
<p><h2>Exploratory Data Analysis</h2>
<h3>Input Data</h3>
<p> Training data provided by Kaggle, consisted of a .csv file with id records for 1216 recordings of birds and frogs with a maximum frequency of 28,000 Hz in flac file format (Figure 1). One audio recording can contain multiple species, but each row has only one label attached to it. In other words, if a species is repeated in the signal, it is included as a separate row in the input .csv file.
</p>
<p><center>
<img src="https://github.com/pnanimal/Portfolio/blob/images/AudioDeepLearning/AudioSample.png" align: center>
</p></center>
<b>Figure 1.</b> Visual representation of one of the audio samples.
<BR><BR>
<P>Two times indicating the beginning and end of the signal were also provided to extract the relevant audio for classification as well as minimum and maximum frequencies.  Species were labeled by integer values between 0-23. Some species had fewer samples than others (Figure 2). Lack of available RAM made augmenting these to the majority class which had 100 samples to be impossible. Rather, species which had less than 50 recordings were augmented to 50 samples.
  </P>  
<p><center>
<img src="https://github.com/pnanimal/Portfolio/blob/images/AudioDeepLearning/Orig_Sampling.png" align: center>
</p></center>
<b>Figure 2.</b> Original sampling of species.
  
