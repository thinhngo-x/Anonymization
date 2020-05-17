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
