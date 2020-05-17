EIGEN=../eigen-eigen-323c052e1731
ANN_INCLUDE=../ann_1.1.2/include
ANN_LIB=../ann_1.1.2/lib
NPY=../libnpy

all: test_dataset test_1q1 test_2q1 test_3q1 test_3q2 test_4q1 test_knn test_random_projection test_roc_curve

Dataset.o: Dataset.cpp ../libnpy/npy.hpp
	g++ -c -I$(EIGEN) -I$(NPY) -Wall -Wextra Dataset.cpp

Classification.o: Classification.cpp
	g++ -c -I$(EIGEN) -Wall -Wextra Classification.cpp
	
KnnClassification.o: KnnClassification.cpp
	g++ -c -I$(EIGEN) -I$(ANN_INCLUDE) -Wall -Wextra KnnClassification.cpp	

ConfusionMatrix.o: ConfusionMatrix.cpp
	g++ -c -I$(EIGEN) -I$(ANN_INCLUDE) -Wall -Wextra ConfusionMatrix.cpp	

RandomProjection.o: RandomProjection.cpp
	g++ -c -I$(EIGEN) -std=c++11 -Wall -Wextra RandomProjection.cpp	

test_dataset: test_dataset.cpp Dataset.o
	g++ -I$(EIGEN) -Wall -Wextra test_dataset.cpp Dataset.o -o test_dataset

test_1q1: test_1q1.cpp KnnClassification.o Classification.o Dataset.o
	g++ -I$(EIGEN) -I$(ANN_INCLUDE) -Wall -Wextra test_1q1.cpp KnnClassification.o Classification.o Dataset.o -o test_1q1 -L$(ANN_LIB) -lANN 

test_2q1: test_2q1.cpp KnnClassification.o Classification.o Dataset.o
	g++ -I$(EIGEN) -I$(ANN_INCLUDE) -Wall -Wextra test_2q1.cpp KnnClassification.o Classification.o Dataset.o -o test_2q1 -L$(ANN_LIB) -lANN 

test_3q1: test_3q1.cpp RandomProjection.o Dataset.o
	g++ -I$(EIGEN) -Wall -Wextra test_3q1.cpp RandomProjection.o Dataset.o -o test_3q1

test_3q2: test_3q2.cpp RandomProjection.o Dataset.o
	g++ -I$(EIGEN) -Wall -Wextra test_3q2.cpp RandomProjection.o Dataset.o -o test_3q2

test_4q1: test_4q1.cpp ConfusionMatrix.o
	g++ -I$(EIGEN) -Wall -Wextra test_4q1.cpp ConfusionMatrix.o -o test_4q1

test_knn: test_knn.cpp ConfusionMatrix.o KnnClassification.o Classification.o Dataset.o
	g++ -I$(EIGEN) -I$(ANN_INCLUDE) -Wall -Wextra test_knn.cpp ConfusionMatrix.o KnnClassification.o Classification.o Dataset.o -o test_knn -L$(ANN_LIB) -lANN 

test_random_projection: test_random_projection.cpp RandomProjection.o ConfusionMatrix.o KnnClassification.o Classification.o Dataset.o
	g++ -I$(EIGEN) -I$(ANN_INCLUDE) -Wall -Wextra test_random_projection.cpp RandomProjection.o ConfusionMatrix.o KnnClassification.o Classification.o Dataset.o -o test_random_projection -L$(ANN_LIB) -lANN

test_roc_curve: test_roc_curve.cpp ConfusionMatrix.o KnnClassification.o Classification.o Dataset.o
	g++ -I$(EIGEN) -I$(ANN_INCLUDE) -Wall -Wextra test_roc_curve.cpp ConfusionMatrix.o KnnClassification.o Classification.o Dataset.o -o test_roc_curve -L$(ANN_LIB) -lANN

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