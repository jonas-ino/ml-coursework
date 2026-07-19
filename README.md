# Machine Learning Coursework
*Archived Coursework: RMIT COSC2673 - Computational Machine Learning*

Three supervised learning projects, each with different problems, focussing on different models.
A common finding across all three projects was that the model choice mattered less than the data. In 2/3, the simplest model won or was tied.
The ceiling was set by the features, rather than the algorithm.

## Stack
Python, scikit-learn, pandas, NumPy, matplotlib, PyTorch

## 1. Airbnb Price Prediction
Task was to predict nightly price for ~8600 listings within Melbourne using 15 features. Focussed on Regression.

Models: Linear, Ridge, and Lasso Regression on raw, standard-scaled, and power-transformed feature sets, looking at the target both raw and log-scaled.

Result: Optimal result was found using Linear Regression, with standard-scaled features, and a log-transformed target (R2 0.43, MAE ~$41).
Ridge and Lasso gave no meaningful improvement. Since there was no dominating coefficient, there was little overfitting to regularise.

Observation: Strongest predictors were number of bedrooms, accommodation capacity, and whether the listing was an entire home.
Linear models aren't able to properly capture non-linear interactions such as longitude/latitude pairings to determine distance to the CBD, which limits potential success.

## 2. Wildfire Intensity Classification
Task was to classify fire intensity (low/moderate/high) from satellite readings. Four ordered classes with an imbalanced target (only 9% extreme). ~4300 rows. Tabular, not image-based.

Models: Decision Tree, SVM (linear and RBF kernels), Neural Network. Evaluated with 5-fold stratified cross validation with 20% held out for the final test to prevent data leakage.

Result: the Decision Tree was the most successful, with a test macro F1 of 0.382, beating the SVM (0.327) and the NN (0.363). Additionally, performed the best with the minority extreme class, which mattered the most.

Observation: Brightness was the only real strong signal from the dataset. The pruned tree was only 3 splits deep, and was still competitive with the other more complex models. Thogh in a small, noisy dataset with weak features, it makes sense that the simplest model would win.
The class imbalance was handled through stratified folds, as underpredicting an extreme fire would be worse than overpredicting one.

## 3. Histopathology Cell Classification
Two tasks observing 27x27 RGB colon cell images from 99 patients.
- isCancerous (binary), cellType (fibroblast/inflammatory/epithelial/others).

Models: Logistic Regression\*, SVM\*, Random Forest\*, Convolutional Neural Network

\* *Models used flattened pixel data* 

Result: CNN won both (isCancerous macro F1: 0.85, cellType macro F1: 0.73). The models using flattened pixel data plateaued below, as they were unable to utilise the spatial structure of the images.
Batch normalisation and data augmentation (flips, rotation, but no brightness jitter, as stain intensity was a key signal) were used to improve the results.

Additional Notes:
- Splits were done per-patient, rather than per-individual image. Cells from one patient would share staining and lighting, so a random split would leak patient data into the test set and potentially inflate the score. Splits keep all patient images on one side, so results reflect unseen patients.
- EDA found that epithelial cells were all cancerous (perfect correlation). An additional dataset was included for the binary task with no cancer classification, which may have impacted overall score.

## Method (shared across all three projects)
- Exploratory Data Analysis
- Data Cleaning
- Pre-processing
- Model training
- Model tuning (using cross-validation to prevent data leakage)
- Transformations  are fit on training folds only and applied to held-out data, so no information leaks from validation or test into the model.

*Datasets not included as they were provided by RMIT*