# Anonymization

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
  - Added few classifiers to the ipynb file "Copy_of_Anonymization_BERT.ipynb"
