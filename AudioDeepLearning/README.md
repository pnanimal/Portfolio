
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

<P>Overall, the ANN had the second best performance with <b>91.5% accuracy</b> on the training, and <b>93.4%</b> accuracy on the test set using a subset of the features. The CNN without transfer learning (augmentation didn’t change the final performance) had the best performance with an accuracy of <b?96.4%</b> on the training, and <b>96.7%</b> on the test set. The RNN using only the MFCC’s achieved an accuracy of <b>96.4%</b> on the training, and <b>93.4%</b> on the test set.
</P>
<p><h2>Exploratory Data Analysis</h2>
<h3>Input Data</h3>
<p> Training data provided by Kaggle, consisted of a .csv file with id records for 1216 recordings of birds and frogs with a maximum frequency of 28,000 Hz in flac file format (Figure 1). One audio recording can contain multiple species, but each row has only one label attached to it. In other words, if a species is repeated in the signal, it is included as a separate row in the input .csv file.
</p>
<p><center>
<img src="https://github.com/pnanimal/Portfolio/blob/images/AudioDeepLearning/AudioSample.png" align: center>
</p></center>
<b>Figure 1. Visual representation of one of the audio samples.</b><BR>
<BR><BR>
<P>Two times indicating the beginning and end of the signal were also provided to extract the relevant audio for classification as well as minimum and maximum frequencies.  Species were labeled by integer values between 0-23. Some species had fewer samples than others (Figure 2). Lack of available RAM made augmenting these to the majority class which had 100 samples to be impossible. Rather, species which had less than 50 recordings were augmented to 50 samples.
  </P>  
<p><center>
<img src="https://github.com/pnanimal/Portfolio/blob/images/AudioDeepLearning/Orig_Sampling.png" align: center>
</p></center>
  <b>Figure 2. Original sampling of species.</b><BR>
<BR><BR><P>
  The data were cut to the min/max times in some cases, and in some cases had additional time subtracted from the start time to increase the total recording time, depending on which choice was randomly selected during augmentation.</p>
  <h2> Data Augmentation</h2><BR>
  <p>The CNN model was evaluated both with and without augmented samples in the training data. <I>Test data were not augmented</I>. Both models achieved the same accuracy, but the model that did not have augmentation converged faster. This result implies that augmentation is not necessary, however the final CNN model uses augmentation with the rationale that the model would be more robust towards unseen data. The ANN and RNN models also used these augmented data in the final models.
	    During augmentation, a target of 50 species was obtained by determining how many times the original audio signal would have to be repeated in order to achieve a total of 50 different recordings for that species. If a number between 1-6 was selected by numpy.random(), the corresponding index from a noise dictionary (Figure 3) was added to the original audio.</p>
<p><center>
<img src="https://github.com/pnanimal/Portfolio/blob/images/AudioDeepLearning/NoiseDict.png" align: center>
</p></center>
<b>Figure 3. Noise dictionary #1 with (upper left) background noise extracted from one of the audio recordings, (upper right) #2 the original noise with random noise added to it, (center left) #3 the original noise with random noise subtracted from it, (center right) is #1 reversed in time, (bottom left) is #2 reversed in time and (bottom right) #3 reversed in time.</b> <BR><BR>
<p>If a number between 7-12 was selected by numpy.random(), then the minimum time was decreased by 1 second, lengthening the audio signal time. If a number between 13-18 is generated, then the audio signal mimics a bird or frog calling twice, an event that does occur in the training data. This process is repeated until each species has at least 50 samples in the training data (Figure 4).</p><br>
<p><center>
<img src="https://github.com/pnanimal/Portfolio/blob/images/AudioDeepLearning/Aug_Sampling.png" align: center>
</p></center>
  <b>Figure 4. Species sampling after augmentation. </b><BR><BR>
<p>After augmentation, the training data was split into train and test data using a 75-25% split with stratification to account for remaining imbalances. The resulting training data was again split into train and validation data using another 75-25% split.</p>
  <h2> Feature Extraction</h2>
  <p>    Audio features had to be prepared differently for input into each of the ANN/CNN/RNN architectures. The following features were extracted using librosa for use in the modeling and modified as described below. The first axis for the input to the CNN corresponds to the audio sample. Spectral bandwidth (Figure 5), chroma spectrogram (Figure 6) and the spectral centroid (Figure 7) were all normalized individually, repeated and padded (Figure 8) and then combined into the second axis in a 4D array (Figure 9). The third axis contained the spectrogram (Figure 10), the fourth contained the MFCCs (Figure 11). Each axis was repeated and padded so that the final 3 slices were identical in size (Figure 12). </p>
<p><center>
<img src="https://github.com/pnanimal/Portfolio/blob/images/AudioDeepLearning/Spec_bw.png" align: center>
</p></center>
<b>Figure 5. Spectral bandwidth computed from the audio file in Figure 1.</b><BR>
<p><center>
<img src="https://github.com/pnanimal/Portfolio/blob/images/AudioDeepLearning/chroma_stft.png" align: center>
</p></center>
<b>Figure 6. Chroma temperature computed from the audio file in Figure 1.</b><BR>
<p><center>
<img src="https://github.com/pnanimal/Portfolio/blob/images/AudioDeepLearning/Spec_centroid.png" align: center>
</p></center>
<b>Figure 7. Spectral centroid computed from the audio file from Figure 1.</b><BR>
<p><center>
<img src="https://github.com/pnanimal/Portfolio/blob/images/AudioDeepLearning/Padding.png" align: center>
</p></center>
<b>Figure 8. The spectral bandwidth (spec_bw), spectral centroid (spec_centroid) and chroma temperature (chroma_stft) were padded, reshaped and repeated to create an axis that was equal in size to the spectrogram in the third axis and the MFCC in the fourth axis.</b>
<p><center>
<img src="https://github.com/pnanimal/Portfolio/blob/images/AudioDeepLearning/FirstAxis.png" align: center>
</p></center>
<b>Figure 9. Scaled second axis of the 4D input into CNN containing padded, reshaped, librosa features.</b?
<p><center>
<img src="https://github.com/pnanimal/Portfolio/blob/images/AudioDeepLearning/spectrogram.png" align: center>
</p></center>
<b>Figure 10. Spectrogram created from the audio displayed in Figure 1.</b>
<p><center>
<img src="https://github.com/pnanimal/Portfolio/blob/images/AudioDeepLearning/MFCC.png" align: center>
</p></center>
<b>Figure 11. MFCCs calculated from the audio displayed in Figure 1.</b>
<p><center>
<img src="https://github.com/pnanimal/Portfolio/blob/images/AudioDeepLearning/ThreeAxis.PNG" align: center>
</p></center>
<b>Figure 12. The four axes for input into the CNN, after normalization, with the first axis corresponding to the audio files (left back), the second corresponding to the padded, reshaped features from Figure 9 (second from back), the spectrogram (second from front) and the MFCCs (front).</b>
<p><h1>Modeling</h1></p>
<p><h2>ANN</h2></p>
<p> The ANN consisted of a layer of 256 hidden nodes connected to the input followed by several additional dense layers with relu activation functions, interspersed with three dropout layers to prevent overfitting the data (Figure 13).  Because an ANN does not preserve spatial information, only the axis corresponding to the padded spectral bandwidth, chroma temperature and spectral centroid were input into the ANN. The spectrogram and MFCCs in particular, have unique features defined spatially. <i>Sending all of the data confused the network and resulted in accuracy levels in the single digits</i>. The net was run using the Adam optimizer for accuracy over 30 epochs. The last layer outputs 24 softmax outputs, corresponding to each of the 24 different species. 
<p><center>
<img src="https://github.com/pnanimal/Portfolio/blob/images/AudioDeepLearning/ANNmodel.png" align: center>
</p></center>
<b>Figure 13. ANN Model architecture.<b><BR><BR>
<p>On the training data, the ANN achieved 91.5% accuracy, and on the test data, it achieved 93.4% accuracy. The confusion matrix (Figure 14) showed that the net had the most problems with species 12, where it misclassified 5 audio files. Species12 was not one of the rarest species nor was it augmented. Further investigation is needed to understand why the network had a problem classifying this particular species.  A perfect result would have predictions for each species only along the diagonal.</p>
<p><center>
<img src="https://github.com/pnanimal/Portfolio/blob/images/AudioDeepLearning/ANNConfusionMatrix.png" align: center>
</p></center>
<b>Figure 14. ANN confusion matrix.</b><BR>
<p>Upon inspecting the training and validation losses and accuracy, the ANN begins to overfit the training data after fifty epochs, but until that point, the net converges to a solution with errors decreasing in the validation data along with the training data (Figure 15).
<p><center>
<img src="https://github.com/pnanimal/Portfolio/blob/images/AudioDeepLearning/ANNCharts.png" align: center>
</p></center>
<b>Figure 15. ANN losses and accuracy by epoch for training and validation datasets.</b><BR>
<h2>CNN</h2>
<p> The CNN takes the input as a 4D numpy array (Figure 12) into a convolutional layer containing 32 hidden layers with relu activation functions (Figure 16). There are three convolutional layers,  two max pooling layers and three dropout layers, and the final layer outputs 24 softmax outputs, corresponding to the 24 species. The Adam optimizer is used over 20 epochs with a loss function to maximize accuracy.  </p>
<p><center>
<img src="https://github.com/pnanimal/Portfolio/blob/images/AudioDeepLearning/CNNmodel.png" align: center>
</p></center>
<b>Figure 16. CNN model architecture.</b><BR><BR>
<p>The confusion matrix (Figure 17) shows that only one sample from species 9 and one from species 19 was misclassified. The accuracy on the training data was 96.4%, and on the test set was 96.7%.  The accuracy and losses behave similarly on both the validation and training sets (Figure 18).
<p><center>
<img src="https://github.com/pnanimal/Portfolio/blob/images/AudioDeepLearning/CNNConfusionMatrix.png" align: center>
</p></center>
<b>Figure 17. CNN confusion matrix.</b><BR>
<p><center>
<img src="https://github.com/pnanimal/Portfolio/blob/images/AudioDeepLearning/CNNCharts.png" align: center>
</p></center>
<b>Figure 18. CNN losses and accuracy by epoch for training and validation datasets.</b><BR>
<h3>Additional CNN testing</h3>
<p>Two more tests were run on the same CNN model. The first was to investigate the model behavior without species augmentation, which converged slightly faster than with augmentation, but to the same degree of accuracy (96.4% on the training and a slightly lower 95.1% on the test set). The final model uses augmentation to improve the robustness of the model for unseen data.</p>
<p><center>
<img src="https://github.com/pnanimal/Portfolio/blob/images/AudioDeepLearning/CNNChartNoAug.png" align: center>
</p></center>
<b>Figure 19. CNN model performance without species augmentation.</b><BR><BR>
<p>The second test was to investigate adding in transfer learning to see if results improved, which they did not. The Oxford VGG16 model, which achieved a high degree of accuracy in the ImageNet classification challenge was used on top of a series of dense layers. The layers of the VGG model were all set to trainable to allow for the adjustments of weights to fit this audio dataset. The training and test data were scaled to fall between 0 and 1 to allow for better integration with the data that was used in VGG16.</p>
<b>Figure 20. CNN model construction with transfer learning from the VGG16 model.</b><BR>
<p>The resulting model had three dense layers with a dropout layer in between after the VGG model in the final architecture (Figure 21).
<p><center>
<img src="https://github.com/pnanimal/Portfolio/blob/images/AudioDeepLearning/CNNmodelTL.png" align: center>
</p></center>
<b>Figure 21. CNN architecture using transfer learning from the VGG16 model.</b><BR>
<p>The model was run for 30 epochs but had slightly poorer performance than the stand-alone CNN model with a training accuracy of 89.5% and test accuracy of 95%. </p>
<h3>RNN</h3>
<p> Similar to the ANN, the RNN needed to have input modified from the CNN 4D tensor. This is because the RNN processes data in a sequence like speech or time sequences. For this reason, the RNN could not understand all of the different kinds of inputs, and performed well by using the axis that contained the MFCC’s (Figure 11) only. The RNN takes the MFCC input into a Long Short-Term Memory (LSTM) layer (Bengio, et. al., 1994) which carries information along a sequence, which in this case corresponds to time. This layer contains 250 hidden layers with relu activation functions (Figure 16). There are five dense layers, and four dropout layers to prevent overfitting the training data. The final layer outputs 24 softmax outputs, corresponding to the 24 species. The Adam optimizer is used over 40 epochs with a loss function to maximize accuracy.  The RNN achieved 94.6% accuracy on the training data and 93.0% accuracy on the test data.</p>
<p><center>
<img src="https://github.com/pnanimal/Portfolio/blob/images/AudioDeepLearning/RNNmodel.pn" align: center>
</p></center>
<b>Figure 22. RNN model architecture.</b><BR><BR>
<p>    The confusion matrix shows that the RNN had the most trouble with species 12 with 9 misclassified audio samples, but had otherwise good performance, considering that, like ANN used only a portion of the data. </p>
<p><center>
<img src="https://github.com/pnanimal/Portfolio/blob/images/AudioDeepLearning/RNNConfusionMatrix.pn" align: center>
</p></center>
<b>Figure 23. RNN confusion matrix.</b><BR><BR>
<p>    The RNN performed better on the validation dataset than it did on the training data, but the loss and accuracy showed that the algorithm converges around 40 epochs (Figure 24).</p>
<p><center>
<img src="https://github.com/pnanimal/Portfolio/blob/images/AudioDeepLearning/RNNCharts.pn" align: center>
</p></center>
<b>Figure 24. RNN losses and accuracy by epoch for training and validation datasets.</b>
<h2>Conclusions and Future Work</h2>
<p>    At an accuracy of 97% on the test set, the CNN model performed the best of all of the deep learning algorithms, however it needed much more data than either the ANN or RNN. If storage capacity and time to extract features are important factors, the ANN had the second best performance with an accuracy of 93.4% accuracy on the test set using only three features: spectral bandwidth, chroma temperature and spectral centroid. The RNN also had a good performance of 87% on the test set using only one feature: the MFCC.
    Future work might consider tailoring the data to exploit the strengths of each deep learning architecture. An ANN might benefit from including data that had little spatial components such as the power spectrum, the spectral rolloff or the audio file itself. The RNN might benefit from having data augmentation on the MFCCs themselves by using different numbers of MFCCs. </p>
<h2>Consulted Resources</h2>
Bengio, Y., P. Simard, and P. Frasconi. "Long short-term memory." IEEE Trans. Neural Netw 5 (1994): 157-166.<BR>
Chollet, F. Deep Learning with Python, 361, New York: Manning, 2018.<BR>
Sarkar, D. Hands on Transfer learning with Python. Retrieved from https://github.com/dipanjanS/hands-on-transfer-learning-with-python on 2/7/21.<BR>
IPython, an interactive shell used to play audio files.<BR>
Joblib, a Python library used to save pickle files.<BR>
Librosa, a Python library for audio analysis.<BR>
Numpy, a Python library for multidimensional matrices and arrays.<BR>
Pandas, a Python data structure with associated libraries.<BR>
pickle, a python library used to convert a byte stream into a storage file.<BR>
os, an operating system interface for Python.<BR>
io, a data extractor that converts data into a structured format.<BR>
Matplotlib, a plotting library in Python.<BR>
Rainforest Connection. (2020). Kaggle Rainforest Connection Species Audio Detection. Retrieved December 30, 2020 from  https://www.kaggle.com/c/rfcx-species-audio-detection.<BR>
Sklearn, a Python machine-learning module.<BR>
Tensorflow, a machine learning platform used for deep learning.<BR>
