#include "ConfusionMatrix.hpp"
#include "LogisticRegression.hpp"
#include "RandomProjection.hpp"
#include "Dataset.hpp"
#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <limits.h>

/** @file
 * Test suite for the LogisticRegression class.
 * This executable will put the two provided CSV files (train and test) in objects of class Dataset, perform logistic regression on the provided column, and print the resulting test set MSE.
*/

int main(int argc, const char * argv[]){
	if (argc < 7) {
        std::cout << "Usage: " << argv[0] << " <projection_dim> <train_file> <train_label> <test_file> <test_label> <sampling> [ <column_for_classification> ]" << std::endl;
        return 1;
    }

	std::cout<< "Reading dataset ..."<<std::endl;

	Dataset train_dataset(argv[2], argv[3],50000);
	Dataset class_dataset(argv[4], argv[5],50000);
    
    int col_class;
    if (argc == 8) {
    	col_class = atoi(argv[7]);
    } else {
		col_class = 0;
		std::cout<< "No column specified for classification, assuming first column of dataset ("<< col_class <<")."<<std::endl;    
    }

	train_dataset.Show(false);  // only dimensions and samples

	assert(train_dataset.getDim() == class_dataset.getDim()); 	// otherwise doesn't make sense
	
	// Tests value of projection_dim (should be > 1 < dimension of dataset)
	int projection_dim=atoi(argv[1]);
	if ((projection_dim<1) | (projection_dim>=train_dataset.getDim()-1)) { 
		std::cout<<"Invalid value of projection_dim."<<std::endl;
		return 1;
	}

	// Tests value of sampling (should be "Gaussian" or "Rademacher")
	std::string sampling=argv[6];
	if ((sampling!="Gaussian") & (sampling!="Rademacher")) { 
		std::cout<<"Invalid value of sampling."<<std::endl;
		return 1;
	}

	// Random projection
    std::cout << "Performing Random Projection" << std::endl;
    clock_t t_random_projection = clock();
	RandomProjection projection(train_dataset.getDim()-1, col_class, projection_dim, sampling);

    t_random_projection = clock() - t_random_projection;
    std::cout << std::endl
         <<"Execution time: "
         <<(t_random_projection*1000)/CLOCKS_PER_SEC
         <<"ms\n\n";
    //projection.ProjectionQuality();

	// Computing logistic regression coefficients on projected data
    std::cout<< "Computing logistic regression coefficients (regression over column "<< col_class << ")..."<<std::endl;
    clock_t t_knn_train_projected = clock();
    Dataset projection_dataset = projection.Project(&train_dataset);
    LogisticRegression tester(&projection_dataset, col_class);
    t_knn_train_projected = clock() - t_knn_train_projected;
    std::cout << std::endl
         <<"Execution time: "
         <<(t_knn_train_projected*1000)/CLOCKS_PER_SEC
         <<"ms\n\n";
	

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
		Dataset projection_test_dataset = projection.Project(&class_dataset);
		clock_t t = clock();
		for (int i=0; i<projection_test_dataset.getNbrSamples(); i++) {
			std::vector<double> sample = projection_test_dataset.getInstance(i);
			Eigen::VectorXd query(projection_test_dataset.getDim()-1);
			double true_label;
			for (int j=0, j2=0; j<projection_test_dataset.getDim()-1 && j2<projection_test_dataset.getDim(); j++, j2++) {
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
		std::cout<< "Prediction" <<std::endl;
		Dataset projection_test_dataset = projection.Project(&class_dataset);
		clock_t t = clock();
		for (int i=0; i<projection_test_dataset.getNbrSamples(); i++) {
			std::vector<double> sample = projection_test_dataset.getInstance(i);
			Eigen::VectorXd query(projection_test_dataset.getDim()-1);
			double true_label;
			for (int j=0, j2=0; j<projection_test_dataset.getDim()-1 && j2<projection_test_dataset.getDim(); j++, j2++) {
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
		error_rate /= projection_test_dataset.getNbrSamples();
		std::cout << "Error rate found: " << error_rate << std::endl;


		t = clock() - t;

		std::cout << std::endl
			<<"execution time: "
			<<(t*1000)/CLOCKS_PER_SEC
			<<"ms\n\n";

	}
    
	return 0;
}