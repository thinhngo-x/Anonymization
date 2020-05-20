# INF442 :  projet informatique 9 - GDPR in practice:  data anonymization

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
    
*Note that convert is just a function to parse the data file into the format of libSVM*

For the subproblem 2,subproblem 3 and subproblem 4, the iPython codes can be found in the folder **Notebook**.
Note that it is required to connect to the following drives, make sure you are connected to it before execute the code:
https://drive.google.com/drive/folders/1KbPiBjhBg70Cy4LgxMa2jOcMCtHQMTKz?usp=sharing
The results can also be found in this drive.

The folder **SVM** is the tools provided by the TD, we can refer to the TD8 for its usage.

You may refer to the pdf for the results.
