<p>
Springboard -- DSC
</p>
<h1><strong>Fatal Encounters Analysis and Modeling</strong></h1>
<p><h2>Papia Nandi
</p>
<p>November 2020
</p>
<p>
<div align="center">
  <img src="https://github.com/pnanimal/FatalEncounters/blob/images/Grave.png"><br><br>
</div>
<h1>Introduction</h1>
<p>
    Fatal Encounters Analysis and Modeling is a study based on the Fatal
Encounters database chronicling people who have died in connection with an
off-duty or on-duty police officers. Modeling and exploratory data are presented as
an attempt to understand the underlying basis of trends, and to establish
connections between discrete fatalities both nationwide and over several
decades.
</p>
<p>
  The analysis is based on data which tabulates deaths from January 1, 2000 to
October 16, 2020, and consists of approximately 30K entries with names, dates,
locations of the deceased, as well as race, gender, age, location and the law
enforcement agency associated with the fatality.
</p>
<p>
  The dataset is assumed to have been recorded and described without significant
error or bias. A Jupyter Notebook contains reproducible code used to explore
some of the trends in gender, cause, and the locality of these fatalities and is
located here:
https://nbviewer.jupyter.org/github/pnanimal/Springboard/blob/master/FatalEncounters.ipynb.
</p>
<p>
  The dataset upon which this analysis is based is available here:
https://docs.google.com/spreadsheets/d/1dKmaV_JiWcG8XBoRgP8b4e9Eopkpgt7FL7nyspvzAsE/edit#gid=0
</p>
<h2>Approach</h2>
<h3>Data Acquisition and Wrangling</h3>
<p>
Data were edited in two parts, the first for exploratory data analysis (EDA)
and the second for modeling. These two steps were separated to preserve the
original data for visualization. To prepare for modeling, the remaining missing
values were filled in using a variety of techniques as described below.
</p>
<p>
<strong>    EDA: </strong>Any missing location data (city, county, zip code) was
imputed by using geolocating code via Nominatum  with latitude and longitude
values recorded in the Fatal Encounters database. After this step, any missing
cities were filled in using data from the injury location. Missing gender
information was filled in using information from the Description information.
Non-numeric information in the Age column (ex: 20’s, 40-50, 3 days) were filled
in using the closest mathematical estimate (20’s->25, 40-50->45, 3 days->3/365).
Zip codes with formats larger than 5 digits were truncated, zip codes that had 4
digits had a ‘0’ filled in front of the value. Injury dates were converted to
datetime format. Comma values from latitude were removed.
</p>
<p>
<strong>    Modeling</strong>: As datetime objects cannot be directly used in
Logistic Regression, the year value was extracted from the Injury date as a
feature for modeling. Missing ages were filled in with the mean value. Missing
gender values, responsible agencies, injury locations were filled in with
‘unknown.’  The imputed race and race columns were combined. Zip code and
latitude values were converted to float. Features were oversampled using
Synthetic Minority Oversampling TEchnique (SMOTE) from Chawla et al., 2002.
</p>
<h3>Exploratory Data Analysis</h3>
<p>
    Several observations can be made with regard to age, gender and cause of
death. Nationwide, deaths peak at an age of 22 years of age with a skewed
distribution towards younger ages (Figure 1). The overwhelming majority of those
encounters are male (90% or 26065) in comparison to female (9.5% or 2752) or
transgender (0.06% or 18). Of these male deaths, the overwhelming majority is
from gunshot (Figure 2), as are deaths for transgender individuals (not shown).
Female deaths are most commonly caused by vehicle accidents.
</p>
<p><center>
<img src="https://github.com/pnanimal/FatalEncounters/blob/images/Age.jpg" align: center>
</p></center>
<p><b>
Figure 1. Histogram of the age and number of deaths from January 1, 2000 to
October 16, 2020 shows a peak in age at 23 years of age.
</p></b>
<p><center>
<img src="https://github.com/pnanimal/FatalEncounters/blob/images/Male_COD.jpg" align: center>
</p></center>
<p>
<b>Figure 2. Cause of death for males shows that gunshot is the most common form.</b>
</p>
Deaths from gunshot occur in every state with the majority of them occurring
in California, followed by Texas, then Florida and Georgia (Figure 3). The
majority of these deaths occur in highly populated areas. In California, for
example, most deaths occurred in and around the Los Angeles area and along
California State Route 99 (Figure 4).
</p>
<center>
<img src="https://github.com/pnanimal/FatalEncounters/blob/images/GunState.png" align="center">
</center>
<p><b>
Figure 3. Distribution of gunshot deaths by state from January 1, 2000 to
October 16, 2020. The largest piece of the pie chart corresponding to 9494
deaths is the total for the remaining 40 states combined.</b>
</p>
<p>
<center>
<img src="https://github.com/pnanimal/FatalEncounters/blob/images/CAGunDeaths.png" align: center>
</p><b></center>
Figure 4. California deaths by gunshot (blue dots)  from January 1, 2000 to
October 16, 2020 show that the majority of deaths were concentrated around large
cities and their surrounding areas and along California State Route 99.
</p></b>
<p>
</p>
<p>
    Nationwide, the number of deaths per year has increased linearly at an
increase of about 50 deaths per year during the two decades over which data were
recorded (Figure 5) by approximately 111% from 2000 to 2019. By comparison, the
population of the United States has increased approximately by only 17% over the
same time frame (United Nations, 2019). These deaths and the increases over time
are primarily from gunshot.
</p>
<p>
    The top-five responsible agencies for these deaths are in descending order
the Los Angeles Police Department (470), the Chicago Police Department (430),
the Los Angeles County Sheriff's Department (350), the City of New York Police
Department (331), and the Houston Police Department (316). However, these
agencies are not responsible for the nationwide increase in gunshot deaths over
the same time period (Figure 6). Moreover, the increased publicity over deaths
and protests associated with them have not appeared to have impacted the daily
number of fatal encounters which occurred afterwards in 2020 (Figure 7).
</p>
<p><center>
<img src="https://github.com/pnanimal/FatalEncounters/blob/images/US_COD.png" class="center">
</p></center>
<p>
<p><b>
Figure 5. Stack plot of nationwide fatal encounters by cause showed that deaths
increased by 111% from January 1, 2000 to December 31, 2019.
</p></b>
<p><center>
<img src="https://github.com/pnanimal/FatalEncounters/blob/images/TopFiveAgenciesbyCOD.png" class="center">
</p></center>
<p><b>
Figure 6. Stack plots of fatal encounters for the (top) Los Angeles Police
Department, (second) Chicago Police Department, (third) Los Angeles County
Sheriff’s Department, (fourth) City of New York Police Department and (bottom)
Houston Police Department show that these individual agencies are not
responsible for the linear trend seen nationwide from 2000 through 2019.</b>
</p>
<p>
<p><center>
<img src="https://github.com/pnanimal/FatalEncounters/blob/images/COD2020.png">
</p></center>
<p><b>
Figure 7. Stack plot deaths nationwide by cause which shows that the number of
deaths in 2020 do not seem to have been impacted by the public deaths of unarmed
Black individuals.</b>
</p>
<p>
    If the top five agencies for fatal encounters are not responsible for the
nationwide increase in gunshot deaths over the same time period, is it possible
to build a model to examine the weights of which features are most important to
an underlying trend? The following sections detail Logistic Regression applied
to predict a fatal encounter by gunshot or otherwise.
</p>
<h3>Baseline Modeling</h3>
<p>
    Logistic regression is a machine learning algorithm that classifies data
into groups. In this study, the linear logistic regression algorithm from
scikit-learn is used to classify data into binary classes with gunshot-related
deaths as a positive (1) result and those associated with other causes as a
negative (0) result. The baseline model used a weighted logistic regression in
conjunction with parameter selection using GridSearchCV with 10 cross-validation
folds to check that the minority class oversampling using SMOTE was sufficient.
Weights were varied between [{0:73,1:27},{0:60,1:40}, {0:50,1:50}, {0:80,1:20},
{0:90,1:10}] based on the initial samping of 73% for the majority class and 27%
for the minority, prior to SMOTE. GridSearchCV was also used to select C, the
inverse of regularization strength from [0.001, 0.1, 1, 10, 100], which acts as
a penalty term for large parameters to avoid overfitting the data.
</p>
<p>
    Results from this baseline model were poor in the test set, with a recall,
defined as the ratio of true positives over the sum of true positive and false
negatives) of only 6% in the minority (oversampled) test set. The predictions
based on this model misclassified 7732 positive results and 132 negative results
(Table 1). The class weights derived from GridSearchCV were split 50-50,
indicating that the results were not negatively influenced by error in the
oversampling technique.
</p>
<p><center>
<img src="https://github.com/pnanimal/FatalEncounters/blob/images/CM_BaseModel.PNG" class="center">
</p></center>
<p>
<b>Table 1. Confusion matrix for the base model shows 7732 misclassified positive
results and 132 negative results for the test case.</b>
</p>
<p>
The classification performance metrics  for the <strong>training</strong>
followed by the <strong>test</strong> set are as follows:
</p>


<pre
class="prettyprint">{'C': 100, 'class_weight': {0: 50, 1: 50}}
              precision    recall  f1-score   support

           0       0.67      0.65      0.66     12217
           1       0.66      0.69      0.68     12351

    accuracy                           0.67     24568
   macro avg       0.67      0.67      0.67     24568
weighted avg       0.67      0.67      0.67     24568

Test Accuracy Score: 0.5199023199023199
Test Area Under Curve: 0.5236661305183202
              precision    recall  f1-score   support

           0       0.80      0.06      0.12      8257
           1       0.51      0.98      0.67      8123

    accuracy                           0.52     16380
   macro avg       0.65      0.52      0.39     16380
weighted avg       0.65      0.52      0.39     16380
</pre>
<h3>Extended Modeling </h3>
<p>
    The quality of the baseline modeling results suggests that the problem may
be more localized. K-Means clustering was able to segregate the data into
similar groups to improve logistic regression modeling results on each
individual cluster. An analysis of the sum of squared error by number of
clusters shows that the data might be optimally split into three clusters
(Figure 8). Logistic regression was then used to model each of the three
clusters individually with a slight improvement in metrics, specifically a
recall of 0.47 on the negative test set for cluster 0, 0.43 for cluster 1, and
0.14 for cluster 2 (Table 2). Although there is marginal improvement over
previous results in the test data, these results suggest that the problem may be
even more localized.
</p>
<p>
<p><center>
<img src="https://github.com/pnanimal/FatalEncounters/blob/images/Clusters.png" class="center">
</p></center>
<p><b>
Figure 8. K-Means clusters by sum of squared distance as a measure of error.</b>
<p>
<p><center>
<img src="https://github.com/pnanimal/FatalEncounters/blob/images/CM_Kmeans.PNG" class="center">
</p></center>
<b>Table 2. Confusion matrices for logistic regression on each of three
clusters derived after K Means clustering, (left) cluster 0/2, (center) cluster
1/2 , (right) cluster 2/2.</b>
<p>
    Modeling at the state level brings slightly better metrics from logistic
regression, depending on the state. In <strong>Texas</strong>, although the
recall score is higher at 0.42 for the negative test set, this is still a
misclassification of more than half of all positive values (Table 3). The
metrics for the training set are better, suggesting that the model is not
generalizing well for the positive case.
</p>
<p>
   Contrasting with Texas, in <strong>California</strong>, the model is having
difficulty predicting the negative case (Table 4), and recall is lower at 31%.
</p>
<p>
<p><center>
<img src="https://github.com/pnanimal/FatalEncounters/blob/images/CM_Texas.PNG" class="center">
</p></center>
<p><b>
Table 3. Confusion matrix for Texas.</b>
</p>
<p>
The classification score for the <strong>training</strong> followed by the
<strong>test</strong> set for <strong>Texas </strong>are as follows:
</p>


<pre
class="prettyprint">{'C': 100, 'class_weight': {0: 50, 1: 50}}
              precision    recall  f1-score   support

           0       0.78      0.79      0.79      1407
           1       0.79      0.78      0.79      1423

    accuracy                           0.79      2830
   macro avg       0.79      0.79      0.79      2830
weighted avg       0.79      0.79      0.79      2830

{'C': 100, 'class_weight': {0: 50, 1: 50}}
Test Accuracy Score: 0.6384180790960452
Test Area Under Curve: 0.6434707629419091
              precision    recall  f1-score   support

           0       0.77      0.42      0.54       362
           1       0.59      0.87      0.70       346

    accuracy                           0.64       708
   macro avg       0.68      0.64      0.62       708
weighted avg       0.68      0.64      0.62       708
</pre>
<p><b>
<p><center>
<img src="https://github.com/pnanimal/FatalEncounters/blob/images/CM_CA.PNG" class="center">
</p></center>
Table 4. Confusion matrix for California.</b>
</p>
<p>
The classification score for the <strong>training</strong> followed by the
<strong>test</strong> set for <strong>California </strong>are as follows:
</p>


<pre
class="prettyprint">{'C': 0.1, 'class_weight': {0: 90, 1: 10}}
              precision    recall  f1-score   support

           0       0.63      1.00      0.77      2659
           1       1.00      0.41      0.58      2656

    accuracy                           0.70      5315
   macro avg       0.81      0.70      0.67      5315
weighted avg       0.81      0.70      0.67      5315

Test Accuracy Score: 0.6245297215951844
Test Area Under Curve: 0.6252462870109928
{'C': 0.1, 'class_weight': {0: 90, 1: 10}}
              precision    recall  f1-score   support

           0       0.58      0.94      0.71       663
           1       0.84      0.31      0.45       666

    accuracy                           0.62      1329
   macro avg       0.71      0.63      0.58      1329
weighted avg       0.71      0.62      0.58      1329
</pre>
<p>

<p>Other states had varying results with <strong>Florida</strong>
at a recall of 0.42 on the positive test set dominated by the number of
incorrectly classified false negatives, <strong>Arizona</strong> with a recall
of 0.51 on the negative test set, <strong>New York</strong> with a recall of
0.10 and <strong>Georgia</strong> with a recall of 0.31 on the positive test
sets.
</p>
<p>
<p><center>
<img src="https://github.com/pnanimal/FatalEncounters/blob/images/CM_States.PNG" class="center">
</p></center>
<b>Table 5. Confusion matrices from left to right: Florida, Arizona, New York, and
Georgia.</b>
</p>
<h3>Findings</h3>
<p>
    Once segregated into states, logistic regression either does poorly at
categorizing the negative test results as in Florida, New York and Georgia or
poorly at predicting the positive test results as in Arizona. Thus, results
should be evaluated on a state by state basis. This is likely because of two
main factors: 1) The negative case lumps several causes together which likely
have many unrelated features, making it difficult to model and 2) the underlying
factors for the positive cases are not captured by the features analyzed.
</p>
<p>
    For the first case, splitting the negative case into the different causes
which include asphyxiation, vehicle, tasered, medical emergencies may not leave
enough samples to model each cause sufficiently and would likely contribute to
undersampling the minority cases even further. For the second case, taking a
deeper look into the descriptions associated with the individual fatalities
using natural language processing may be necessary. Thus, only the largest
weights from each state should be considered to be important features, and even
these should be understood to contain a significant amount of error in
predicting the negative case, depending on the model metrics.
</p>
<p>
    A summary of the model results are given in Table 6. with the <strong>Test
Area under the curve (AUC)</strong> metric defined as the probability that a
classifier will rank a randomly chosen positive instance higher than a randomly
chosen negative one (Fawcett, 2006), and <strong>Test Accuracy</strong> defined
as the proportion of correct predictions to samples.
</p>
<p>
<p><center>
<img src="https://github.com/pnanimal/FatalEncounters/blob/images/ScoreTable.png" class="center">
</p></center>
<p>
<b>Table 6. Summary table for Test Accuracy and AUC score, for all modeled
scenarios.</b>
</p>
<p>
    These metrics for model quality are important, but not the ultimate goal for
this analysis, which is to measure feature importance to understand regional
underlying factors for encounters that end in death by gunshot or otherwise. The
feature importance is related to the magnitudes and values of the weights. A
large positive weight indicates that the feature has a large contribution
towards predicting a positive (gunshot) death. A large negative weight indicates
that the feature has a large contribution towards predicting the negative (not
gunshot-related) death. A small weight (either positive or negative) indicates
that the feature is not very important and should be considered as noise.
</p>
<p>
   In <strong>Florida</strong>, the highest positive weight corresponds to the
gender of male followed by year, suggesting that different from either Texas or
California gunshot deaths are not weighted heavily by gender and also changing
with time (Figure 11). Similar to <strong>Georgia</strong> and<strong> New
York</strong>, Florida has its highest positive weight associated with the male
gender. This strong factor is likely because the overwhelming majority of males
that have fatal encounters die by gunshot, and these deaths are the majority
class for each state. After the highest weights, each of the largest factors
vary per state between locales (city, county) and responsible agencies.
</p>
<p>
    In <strong>Texas</strong>, the highest positive weights are associated with
the border and Fort Worth and surrounding area. The largest negative weight is a
combination of two responsible agencies, and should be neglected for analysis,
but the remaining negative weights are cities with low populations (Colorado and
Columbus). That the city of Brookshire and the Brookshire Police departments
appear as weights with opposite signs, likely indicates the limit of the
modeling as noise, and suggests that only the largest weights should be
considered to be significant, and that the results should be interpreted
appropriately.
</p>
<p>
<p><center>
<img src="https://github.com/pnanimal/FatalEncounters/blob/images/TXWeights.png" class="center">
</p></center>
<p>
<b>Figure 9. Top ten positive and negative Texas Logistic regression weights scaled
by standard deviation.</b>
</p>
<p>
    In <strong>California</strong>, results show that the highest positive
weight corresponds to the city of El Monte, an industrial city outside of Los
Angeles, but that the second highest weight corresponds to the city of Anderson,
a small city near Redding, which indicates that Redding may warrant further
examination to find out why so many deaths by gunshot are taking place here. In
fact, Redding and its surrounding area has been the subject of scrutiny for
having an abnormally high rate of officer-related fatalities: <a
href="https://www.redding.com/story/news/local/2019/12/18/officer-involved-shootings-shasta-county-last-decade/2502112001/">https://www.redding.com/story/news/local/2019/12/18/officer-involved-shootings-shasta-county-last-decade/2502112001/</a>.
The next highest weight is for the city of Beaumont appears, whose associated
police department also appears as a negative weight, indicating that this
feature and those with smaller weights fall within the level of noise and should
be discounted.
</p>
<p>
<p><center>
<img src="https://github.com/pnanimal/FatalEncounters/blob/images/CAWeights.png" class="center">
</p></center>
<p>
<b>Figure 10. Top ten positive and negative California Logistic regression weights,
scaled by standard deviation.</b>
</p>
<p>
    In <strong>Arizona</strong>, results were different than any of the other
analyzed states because it has very low positive weights (Figure 11). Arizona
also contained very high negative weights, the largest of which corresponded to
the race Asian/Pacific Islander, which did not appear as one of the ten most
positive or negative weights in any other state analyzed. Several of the
highest negative weights were associated with Navajo County and the city of
Holbrook which lies within it.
</p>
<p>
<p><center>
<img src="https://github.com/pnanimal/FatalEncounters/blob/images/AZWeights.png" class="center">
</p></center>
<p>
<b>Figure 11. Top ten positive and negative Arizona Logistic regression weights,
scaled by standard deviation.</b>
</p>
<p>
    In <strong>Florida</strong>, the highest positive weight corresponds to the
gender of male followed by year, suggesting that differently from either Texas
or California that gunshot deaths are weighted heavily by gender and also
changing with time (Figure 12). The next two highest positive weights for the
Davie Police Department and the Palm Beach County Sheriff’s Department are
physically located next to each other on Florida’s popular beach coastline.
</p>
<p>
<p><center>
<img src="https://github.com/pnanimal/FatalEncounters/blob/images/FLWeights.png" class="center">
</p></center>
<p>
<b>Figure 12. Top ten positive and negative Florida Logistic regression weights,
scaled by standard deviation.</b>
</p>
<p>
    Both <strong>New York</strong> and <strong>Georgia</strong> had particularly
low recall rates in the positive test set of 0.10. In New York, the highest
positive score corresponds to the male gender, and the second highest to Nassau
County, home to Long Island (Figure 13). Georgia also had the highest positive
weight corresponding to the male gender, with the second being the race of
Asian/Pacific Islander (Figure 14). The highest negative weight corresponds to
the Georgia State Patrol.
</p>
<p>
<p><center>
<img src="https://github.com/pnanimal/FatalEncounters/blob/images/NYWeights.png" class="center">
</p></center>
<p>
<b>Figure 13. Top ten positive and negative New York Logistic regression weights,
scaled by standard deviation.</b>
</p>
<p>
<p><center>
<img src="https://github.com/pnanimal/FatalEncounters/blob/images/GAWeights.png" class="center">
</p></center>
<p><b>
Figure 14. Top ten positive and negative Georgia Logistic regression weights,
scaled by standard deviation.</b>
</p>
<h3>Conclusions and Future Work</h3>
<p>
   Though many insights have been revealed by the exploratory data analysis,
modeling to find underlying weights is a complex task. Each state should be
evaluated separately and interpreted in accordance with the errors produced in
the test set in mind. Given additional time and sources, the following future
work may lead to additional insights:
</p>
<ul>
<li>Create models by using even more localized data at a county or city level.
<li>Use natural language processing techniques to analyze the information
contained in the descriptions provided for each fatal encounter.
<li>Use time series analysis to investigate the increasing trend of
gunshot-related deaths over the past two decades to predict a future trend.
<li>Find which weights are contributing to the increase in fatal encounters
nationally.
<li>In addition to TX, GA, CA, NY, AZ and FL, expand the modeling to other
states.
</li>
</ul>
<h3>Recommendations for the Client</h3>
<p>
    Fatal encounters are increasing nationally, driven mostly by gunshot deaths
after encounters with police that are not associated with the agencies which
have the highest number of deaths associated with them. These findings can be
used on a regional level to understand and investigate, through specific models,
 important features that contribute to gunshot-related deaths, the fastest
increasing component of fatal encounters.
</p>
<p>
   In Texas, for example, the highest weights are associated with the city of
Fort Worth and its surrounding area, yet Fort Worth is not the most populated
city in Texas. Such discrepancies can be explored to find what interventions are
made in Houston, for example, to prevent gunshot-related fatalities elsewhere in
the state if appropriate and desired.
</p>
<p>
   In Florida, many of the fatalities are associated with beach and resort
areas. Can this area become a focus for prevention? The type of modeling
exemplified in this project shows how insights from models can be used to focus
efforts and increase community understanding of the underlying problems that
could lead to fatal encounters and how to start reducing them over time.
</p>
<ul>
<li>Use the modeling to focus on discrepancies between similar areas in
population or location that have different feature weights.
<li>Understand regional behaviors and how they affect national trends.
<li>Plan and budget resources towards reducing discrepancies where appropriate.
</li>
</ul>
<h3>Consulted Resources</h3>
<p>
<strong>United Nations World Population Prospects</strong>, <a
href="https://population.un.org/wpp/">https://population.un.org/wpp/</a>
</p>
<p>
<strong>Fawcett, Tom</strong> (2006); An introduction to ROC analysis, Pattern
Recognition Letters, 27, 861–874.
</p>
<p>
<strong>Chawla, N. V., Bowyer, K. W., Hall., L. O., Kegelmeyer, W. P</strong>.,
2002, SMOTE: Synthetic Minority Over-sampling Technique: Journal Of Artificial
Intelligence Research, Volume 16, pages 321-357.
</p>
<p>
<strong>Contextily</strong>, a Python package to retrieve tile maps for
map-based plotting.
</p>
<p>
<strong>Datetime</strong>, a Python module for manipulating dates and times.
</p>
<p>
<strong>Geopandas</strong>, an extension of Pandas that allows mathematical
operations on geotypes.
</p>
<p>
<strong>Geopy</strong>, a Python geocoding library.
</p>
<p>
<strong>Google colab</strong>, a cloud-based Jupyter notebook.
</p>
<p>
<strong>Imbalance-learn</strong>, a Python package for correcting sampling
differences between a less sampled minority class and majority class from which
Synthetic Minority Oversampling TEchnique (<strong>SMOTE)</strong> was used.
</p>
<p>
<strong>io</strong>, a data extractor that converts data into a structured
format.
</p>
<p>
<strong>Matplotlib</strong>, a plotting library in Python.
</p>
<p>
<strong>Nominatim</strong>, a tool for reverse geocoding by address.
</p>
<p>
<strong>Numpy</strong>, a Python library for multidimensional matrices and
arrays.
</p>
<p>
<strong>OpenStreetMap</strong>, a global source of map data.
</p>
<p>
<strong>os</strong>, an operating system interface for Python.
</p>
<p>
<strong>Pandas</strong>, a Python data structure with associated libraries.
</p>
<p>
<strong>Pygal</strong>, a Python module for creating scalable vector graphics
(SVG).
</p>
<p>
<strong>Seaborn</strong>, a Python data visualization package based.
</p>
<p>
<strong>Sklearn</strong>, a Python machine-learning module that including
<strong>metrics</strong>, a library for calculating different types of
mathematically based performance scores for models from which
<strong>classification_report, confusion_matrix, roc_auc_score</strong> and
<strong>accuracy score</strong> were used,, <strong>model_selection</strong>, a
library containing parameterization modules for modeling from which<strong>
train_test_split, GridSearchCV</strong> and <strong>KFold</strong> were used,
<strong>feature_selection</strong>, a module used in dimensionality reduction
from which <strong>SelectKBest</strong> was used,
<strong>decomposition</strong>, a module to reduce dimensionality and project
onto a lower-dimensional space from which<strong> PCA</strong> was used,
<strong>linear_model</strong>, a module for linear modeling from which
<strong>LogisticRegression</strong> was used, <strong>preprocessing</strong>, a
module for preparing data for modeling from which <strong>scale</strong> and
<strong>StandardScaler</strong> were used, and <strong>cluster</strong>, a
module for grouping data using machine learning from which
<strong>KMeans</strong> was used.
</p>
