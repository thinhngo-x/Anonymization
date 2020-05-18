EIGEN=../../Background/eigen-eigen-323c052e1731
ANN_INCLUDE=../../Background/ann_1.1.2/include
ANN_LIB=../../Background/ann_1.1.2/lib
NPY=../../Background/libnpy

all: test_dataset test_knn test_random_projection test_roc_curve test_linear_regression test_logistic_regression test_linear_regression_random test_logistic_regression_random convert

convert: convert.o
	cc -g -o convert convert.o

convert.o: convert.c
	cc -c -Wall -g convert.c

Dataset.o: Dataset.cpp
	g++ -c -I$(EIGEN) -I$(NPY) -Wall -Wextra Dataset.cpp

Classification.o: Classification.cpp
	g++ -c -I$(EIGEN) -Wall -Wextra Classification.cpp
	
KnnClassification.o: KnnClassification.cpp
	g++ -c -I$(EIGEN) -I$(ANN_INCLUDE) -Wall -Wextra KnnClassification.cpp	

ConfusionMatrix.o: ConfusionMatrix.cpp
	g++ -c -I$(EIGEN) -I$(ANN_INCLUDE) -Wall -Wextra ConfusionMatrix.cpp	

RandomProjection.o: RandomProjection.cpp
	g++ -c -I$(EIGEN) -std=c++11 -Wall -Wextra RandomProjection.cpp	

LinearRegression.o: LinearRegression.cpp
	g++ -c -I$(EIGEN) -Wall -Wextra LinearRegression.cpp

LogisticRegression.o: LogisticRegression.cpp
	g++ -c -I$(EIGEN) -Wall -Wextra LogisticRegression.cpp

test_dataset: test_dataset.cpp Dataset.o
	g++ -I$(EIGEN) -Wall -Wextra test_dataset.cpp Dataset.o -o test_dataset

test_knn: test_knn.cpp ConfusionMatrix.o KnnClassification.o Classification.o Dataset.o
	g++ -I$(EIGEN) -I$(ANN_INCLUDE) -Wall -Wextra test_knn.cpp ConfusionMatrix.o KnnClassification.o Classification.o Dataset.o -o test_knn -L$(ANN_LIB) -lANN 

test_random_projection: test_random_projection.cpp RandomProjection.o ConfusionMatrix.o KnnClassification.o Classification.o Dataset.o
	g++ -I$(EIGEN) -I$(ANN_INCLUDE) -Wall -Wextra test_random_projection.cpp RandomProjection.o ConfusionMatrix.o KnnClassification.o Classification.o Dataset.o -o test_random_projection -L$(ANN_LIB) -lANN

test_roc_curve: test_roc_curve.cpp ConfusionMatrix.o KnnClassification.o Classification.o Dataset.o
	g++ -I$(EIGEN) -I$(ANN_INCLUDE) -Wall -Wextra test_roc_curve.cpp ConfusionMatrix.o KnnClassification.o Classification.o Dataset.o -o test_roc_curve -L$(ANN_LIB) -lANN

test_linear_regression: test_linear_regression.cpp ConfusionMatrix.o LinearRegression.o Classification.o Dataset.o
	g++ -I$(EIGEN) -Wall -Wextra test_linear_regression.cpp ConfusionMatrix.o LinearRegression.o Classification.o Dataset.o -o test_linear_regression 

test_logistic_regression: test_logistic_regression.cpp ConfusionMatrix.o LogisticRegression.o Classification.o Dataset.o
	g++ -I$(EIGEN) -Wall -Wextra test_logistic_regression.cpp ConfusionMatrix.o LogisticRegression.o Classification.o Dataset.o -o test_logistic_regression

test_linear_regression_random: test_linear_regression_random.cpp RandomProjection.o ConfusionMatrix.o LinearRegression.o Classification.o Dataset.o
	g++ -I$(EIGEN) -Wall -Wextra test_linear_regression_random.cpp RandomProjection.o ConfusionMatrix.o LinearRegression.o Classification.o Dataset.o -o test_linear_regression_random 

test_logistic_regression_random: test_logistic_regression_random.cpp RandomProjection.o ConfusionMatrix.o LogisticRegression.o Classification.o Dataset.o
	g++ -I$(EIGEN) -Wall -Wextra test_logistic_regression_random.cpp RandomProjection.o ConfusionMatrix.o LogisticRegression.o Classification.o Dataset.o -o test_logistic_regression_random

.PHONY: all clean

clean:
	rm -f *.o
	rm -f test_dataset
	rm -f test_1q1
	rm -f test_2q1
	rm -f test_3q1
	rm -f test_3q2
	rm -f test_4q1
	rm -f test_knn
	rm -f test_random_projection
	rm -f test_5q1
	rm -f test_roc_curve
	rm -f test_linear_regression
	rm -f test_logistic_regression