#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include "Dataset.hpp"
#include "RandomProjection.hpp"
#include "KnnClassification.hpp"
#include "ConfusionMatrix.hpp"
#include <cstdlib>
using namespace std;

/** @file
*/

// Prints correct usage
void msgleave(char* argv[]) {
        std::cout << "Usage: " << argv[0] << " <k> <projection_dim> <train_file> <true_labels.train> <test_file> <true_labels.test> <sampling> [ <column_for_classification> ]" << std::endl;
}

int main(int argc, char* argv[]) {
    // Tests correct usage
	if (argc < 6) {
		msgleave(argv);
		return 1;
	}
		
    // Puts train file in a Dataset object
	Dataset train_dataset(argv[3], argv[4]);
	Dataset class_dataset(argv[5], argv[6]);

    // Tests value of projection_dim (should be > 1 < dimension of dataset)
	int projection_dim=atoi(argv[2]);
	if ((projection_dim<1) | (projection_dim>=train_dataset.getDim()-1)) { 
		std::cout<<"Invalid value of projection_dim."<<std::endl;
		msgleave(argv);
		return 1;
	}

    // Tests value of k (should be > 1)
	int k=atoi(argv[1]);
	if (k<1) { 
		std::cout<<"Invalid value of k."<<std::endl;
		msgleave(argv);
		return 1;
	}

    // Tests value of sampling (should be "Gaussian" or "Rademacher")
	std::string sampling=argv[7];
	if ((sampling!="Gaussian") & (sampling!="Rademacher")) { 
		std::cout<<"Invalid value of sampling."<<std::endl;
		msgleave(argv);
		return 1;
	}
    
    // Checks which column is the class label
    int col_class;
    if (argc == 9) {
    	col_class = atoi(argv[8]);
    } else {
		//col_class = train_dataset.getDim()-1;
        col_class = 0;
		std::cout << "No column specified for classification, assuming first column of dataset ("<< col_class <<")."<<std::endl;
    }

    // Prints dimension
	train_dataset.Show(false);  // only dim and samples

	// Random projection
    std::cout << "Performing Random Projection" << std::endl;
    clock_t t_random_projection = clock();
	RandomProjection projection(train_dataset.getDim()-1, col_class, projection_dim, sampling);

    t_random_projection = clock() - t_random_projection;
    std::cout << endl
         <<"Execution time: "
         <<(t_random_projection*1000)/CLOCKS_PER_SEC
         <<"ms\n\n";
    //projection.ProjectionQuality();

    // Performing Knn on original data
    // std::cout << "Performing Knn on original data" << std::endl;
    // clock_t t_knn_train_original = clock();
    // KnnClassification knn_class_original(k, &train_dataset, col_class);
    // t_knn_train_original = clock() - t_knn_train_original;
    // std::cout << endl
    //      <<"Execution time: "
    //      <<(t_knn_train_original*1000)/CLOCKS_PER_SEC
    //      <<"ms\n\n";

    // Performing Knn on projected data
    std::cout << "Performing Knn on projected data" << std::endl;
    clock_t t_knn_train_projected = clock();
    Dataset projection_dataset = projection.Project(&train_dataset);
    KnnClassification knn_class_projected(k, &projection_dataset, 0);
    t_knn_train_projected = clock() - t_knn_train_projected;
    std::cout << endl
         <<"Execution time: "
         <<(t_knn_train_projected*1000)/CLOCKS_PER_SEC
         <<"ms\n\n";

    // Knn test on original data
    // std::cout << "Predicting Knn on original data" << std::endl;
    // ConfusionMatrix confusion_matrix_original;
    // clock_t t_knn_test_original = clock();
    // for (int i=0; i<class_dataset.getNbrSamples(); i++) {
    //     std::vector<double> sample = class_dataset.getInstance(i);
    //     Eigen::VectorXd query(class_dataset.getDim()-1);
    //     int true_label;
    //     for (int j=0, j2=0; j<train_dataset.getDim()-1 && j2<train_dataset.getDim(); j++, j2++) {
    //         if (j==col_class && j2==col_class) {
    //             true_label = sample[j2];
    //             j--;
    //             continue;
    //         }
    //         query[j] = sample[j2];
    //     }
    //     int predicted_label = knn_class_original.Estimate(query);
    //     confusion_matrix_original.AddPrediction(true_label, predicted_label);
    // }
    // t_knn_test_original = clock() - t_knn_test_original;
    // std::cout << endl
    //      <<"Execution time: "
    //      <<(t_knn_test_original*1000)/CLOCKS_PER_SEC
    //      <<"ms\n\n";
    // confusion_matrix_original.PrintEvaluation();

    // Knn test on projected data
    std::cout << "Predicting Knn on projected data" << std::endl;
    ConfusionMatrix confusion_matrix_projected;
    Dataset projection_test_dataset = projection.Project(&class_dataset);
    clock_t t_knn_test_projected = clock();
    for (int i=0; i<projection_test_dataset.getNbrSamples(); i++) {
        std::vector<double> sample = projection_test_dataset.getInstance(i);
        Eigen::VectorXd query(projection_test_dataset.getDim()-1);
        // std::cout<<projection.getProjectionDim()<<std::endl;
        int true_label;
        // std::cout<<"YES"<<std::endl;
        for (int j=0, j2=0; j<projection_test_dataset.getDim()-1 && j2<projection_test_dataset.getDim(); j++, j2++) {
            if (j==col_class && j2==0) {
                true_label = sample[j2];
                j--;
                // std::cout<<j<<std::endl;
                continue;
            }
            // std::cout<<"YES"<<std::endl;
            query[j] = sample[j2];
            // std::cout<<j<<std::endl;
        }
        // std::cout<<query[9]<<std::endl;
        int predicted_label = knn_class_projected.Estimate(query);
        confusion_matrix_projected.AddPrediction(true_label, predicted_label);
        // std::cout<<"YES"<<std::endl;
    }
    t_knn_test_projected = clock() - t_knn_test_projected;
    std::cout << endl
         <<"Execution time: "
         <<(t_knn_test_projected*1000)/CLOCKS_PER_SEC
         <<"ms\n\n";
    confusion_matrix_projected.PrintEvaluation();

    // Speed up
    // std::cout << "Speedup on training: " << t_knn_train_original / t_knn_train_projected << std::endl;
    
    // std::cout << "Speedup on testing: " << t_knn_test_original / t_knn_test_projected << std::endl;

}