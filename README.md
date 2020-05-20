# Anonymization

## Usage
The folders **ann_1.1.2**, **eigen-eigen-323c052e1731** and **libnpy** are the useful library for C++.
For the subproblem 1 and subproblem 3, simply use "make" in the folder **code of C++**.
The functions of each executable are as follows: (for the necessary arguments, you can simply execute the function and follow the instructions) 
  - test_dataset
    - similar to TD, but the method to load the data from .npy is added
  - test_knn
    - perform kNN classification to the dataset
  - test_random_projection
    - perform kNN classification to the projected dataset
  - test_linear_regression
    - perform linear regression for classification to the dataset
  - test_linear_regression_random
    - perform linear regression for classification to the projected dataset
  - test_logistic_regression
    - perform logistic regression for classification to the dataset
  - test_logistic_regression_random
    - perform logistic regression for classification to the projected dataset
    
*note that convert is just a function to parse the data file into the format of libSVM*

For the subproblem 2,subproblem 3 and subproblem 4, the iPython codes can be found in the folder **Notebook**.
Note that it is required to connect to the following drives, make sure you are connected to it before execute the code:
https://drive.google.com/drive/folders/1KbPiBjhBg70Cy4LgxMa2jOcMCtHQMTKz?usp=sharing
The results can also be found in this drive.

The folder **SVM** is the tools provided by the TD, we can refer to the TD8 for its usage.

## Changelog
17/05 18:44 Thinh
  - I've run the notebook to re-produce the dataset. The true_labels files are changed to number-type, which allow C++ to read it easily with libnpy.
  - This version only works with .npy files.
  - Because loading too much data would take time and space to store, so I set a variable whose name is *maxiter* (in the class Dataset.hpp) to indicate the number of samples we'd like to take for the test. You could try first with ./test_dataset while seting *maxiter* from 10000 to see how many samples your computers could deal with.
  - Don't run ./test_knn if you don't want to risk your computer hanged in the middle.
  - On my computer, by setting *maxiter* to 50000, I ran ./test_random_projection with k = 5, projection_dim = 10, on the testb, and finished it in about 7 minutes.
  - There is a problem with the representation.testa.npy so the testa won't work.
  - Remember the labels : {0 : "O", 1 : "B-MISC", 2 : "I-MISC", 3 : "B-PER",
                           4 : "I-PER", 5 : "B-ORG", 6 : "I-ORG", 7 : "B-LOC", 8 : "I-LOC"}

18/05 15:15 Chin Wei
  - I have downloaded the files in github but unable to run even the testdata code due to an unknown reason, will recitify it later. (even change the maxiter)
  - Added some files related to linear and logistic regression
  - I checked the libSVM provided by the TD, extract the executable and find a converter (written in C). The usage as below:
      - run ./convert [csv_file] >> .train/.test to convert to svm file format 
      - run ./svm-train [training file] ==> the model file is generated and the details of the relevant parameters are displayed
      - run ./svm-predict [testing file] [training file].model [output file] ==> the accuracy is displayed and the predicted class labels can be found in output file
      - ***can use options too for different kernel
  - Added few classifiers to the ipynb file "Copy_of_Anonymization_BERT.ipynb"
