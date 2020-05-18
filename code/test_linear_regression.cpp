#include "ConfusionMatrix.hpp"
#include "LinearRegression.hpp"
#include "Dataset.hpp"
#include <iostream>
#include <cstdlib>
#include <stdio.h>

/** @file
 * Test suite for the LinearRegression class.
 * This executable will put the two provided CSV files (train and test) in objects of class Dataset, perform linear regression on the provided column, and print the resulting test set MSE.
*/

int main(int argc, const char * argv[]){
	if (argc < 6) {
        std::cout << "Usage: " << argv[0] << " <k> <train_file> <train_label> <test_file> <test_label> [ <column_for_classification> ]" << std::endl;
        return 1;
    }

	std::cout<< "Reading training dataset "<<argv[1]<<"..."<<std::endl;

	Dataset train_dataset(argv[2], argv[3]);
	Dataset class_dataset(argv[4], argv[5]);
    
    int col_class;
    if (argc == 4) {
    	col_class = atoi(argv[3]);
    } else {
		col_class = 0;
		std::cout<< "No column specified for classification, assuming first column of dataset ("<< col_class <<")."<<std::endl;    
    }

	train_dataset.Show(false);  // only dimensions and samples

	assert(train_dataset.getDim() == class_dataset.getDim()); 	// otherwise doesn't make sense
	
	std::cout<< "Computing linear regression coefficients (regression over column "<< col_class << ")..."<<std::endl;
	LinearRegression tester(&train_dataset, col_class);

	tester.ShowCoefficients();


	std::cout<< "Testing the Estimate method on the first sample of test_file" <<std::endl;
	std::vector<double> first_sample = class_dataset.getInstance(0);
	Eigen::VectorXd first_sample_eigen(class_dataset.getDim() - 1);

	for (int j = 0, j2 = 0; j < class_dataset.getDim() - 1 && j2 < class_dataset.getDim(); j++, j2++) {
		if (j==col_class && j2==col_class) {
			j--;
			continue;
		}
		first_sample_eigen(j) = first_sample[j2];
	}

	std::cout<< "The true value of y for the first sample of test_file is:" <<std::endl;
	std::cout<< first_sample[col_class] <<std::endl;
	std::cout<< "The estimated value of y for the first sample of test_file is:" <<std::endl;
	std::cout<< tester.Estimate(first_sample_eigen) <<std::endl;

	if (tester.getClassNo() == 2){

		// ConfusionMatrix
		ConfusionMatrix confusion_matrix;

		// Starts predicting
		std::cout<< "Prediction and Confusion Matrix filling" <<std::endl;
		clock_t t = clock();
		for (int i=0; i<class_dataset.getNbrSamples(); i++) {
			std::vector<double> sample = class_dataset.getInstance(i);
			Eigen::VectorXd query(class_dataset.getDim()-1);
			double true_label;
			for (int j=0, j2=0; j<train_dataset.getDim()-1 && j2<train_dataset.getDim(); j++, j2++) {
				if (j==col_class && j2==col_class) {
					true_label = sample[j2];
					j--;
					continue;
				}
				query(j) = sample[j2];
			}
			double predicted_label = tester.Estimate(query);
			confusion_matrix.AddPrediction(true_label, predicted_label);
		}


		t = clock() - t;

		std::cout << std::endl
			<<"execution time: "
			<<(t*1000)/CLOCKS_PER_SEC
			<<"ms\n\n";

		confusion_matrix.PrintEvaluation();

	}
	else{
		// Starts predicting
		double error_rate = 0.0;
		std::cout<< "Prediction and Confusion Matrix filling" <<std::endl;
		clock_t t = clock();
		for (int i=0; i<class_dataset.getNbrSamples(); i++) {
			std::vector<double> sample = class_dataset.getInstance(i);
			Eigen::VectorXd query(class_dataset.getDim()-1);
			double true_label;
			for (int j=0, j2=0; j<train_dataset.getDim()-1 && j2<train_dataset.getDim(); j++, j2++) {
				if (j==col_class && j2==col_class) {
					true_label = sample[j2];
					j--;
					continue;
				}
				query(j) = sample[j2];
			}
			double predicted_label = tester.Estimate(query);
			if (true_label != predicted_label){
				error_rate++;
			}
		}
		error_rate /= class_dataset.getNbrSamples();
		std::cout << "Error rate found: " << error_rate << std::endl;


		t = clock() - t;

		std::cout << std::endl
			<<"execution time: "
			<<(t*1000)/CLOCKS_PER_SEC
			<<"ms\n\n";

	}
    
	return 0;
}