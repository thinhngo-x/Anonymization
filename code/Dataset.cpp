
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cassert>
#include "Dataset.hpp"
#include "../libnpy/npy.hpp"

int Dataset::getNbrSamples() {
	return m_nsamples;
}

int Dataset::getDim() {
	return m_dim;
}

Dataset::~Dataset() {
	// All attributes have destructors
}

void Dataset::Show(bool verbose) {
	std::cout<<"Dataset with "<<m_nsamples<<" samples, and "<<m_dim<<" dimensions."<<std::endl;
	if (verbose) {
		for (int i=0; i<10; i++) {
			for (int j=0; j<10; j++) {
				std::cout<<m_instances[i][j]<<" ";
			}
			std::cout<<std::endl;		
		}
	}
}

Dataset::Dataset(const char* file)
{
	std::cout<<"Warning: You're reading only the dataset"<<std::endl;
	std::cout<<"Add the path to true_labels file into the parameters to read true labels."<<std::endl;
	
	std::vector<unsigned long> shape;
	bool fortran_order;
	std::vector<float> data;
	auto path = file;

	shape.clear();
	data.clear();
	npy::LoadArrayFromNumpy(path, shape, fortran_order, data);

	m_nsamples = (int) shape[0];
	// std::cout<<m_nsamples<<std::endl;
	m_dim = (int) shape[1];

	std::cout << "shape: ";
	for (size_t i = 0; i<shape.size(); i++)
		std::cout << shape[i] << ", ";
	std::cout << std::endl;
	std::cout << "fortran order: " << (fortran_order ? "+" : "-");
	std::cout << std::endl;

	for(int i = 0; i<m_nsamples; i++)
	{	
		if(i % 10000 == 0)
			std::cout<<"Reading..."<<i<<std::endl;
		std::vector<double> row;
		for(int j = 0; j<m_dim; j++)
		{
			int index = i*m_dim + j;
			row.push_back((double) data[index]);
		}
		// if(i == 203592)
		// 	std::cout<<row.size()<<std::endl;
		m_instances.push_back(row);
	}
}


Dataset::Dataset(const char* file, const char* true_labels)
{

	auto path = true_labels;
	std::vector<unsigned long>shape;
	bool fortran_order;
	std::vector<long> labels;
	
	shape.clear();
	labels.clear();
	std::cout<<"Loading true_labels ..."<<std::endl;
	npy::LoadArrayFromNumpy(path, shape, fortran_order, labels);

	m_nsamples = shape[0];
	for (auto i = 0; i<std::min(m_nsamples, maxiter); i++)
	{
		std::vector<double> v;
		if(labels[i] == 3 || labels[i] == 4)
			v.push_back(1.0);
		else
			v.push_back(0.0);
		m_instances.push_back(v);
	}
	
	m_dim = 1;
	labels.clear();
	labels.shrink_to_fit();


	std::vector<float> data;
	path = file;
	
	std::cout<<"Loading representation ..."<<std::endl;
	shape.clear();
	data.clear();
	npy::LoadArrayFromNumpy(path, shape, fortran_order, data);

	m_nsamples = std::min((int) shape[0], maxiter);
	// std::cout<<m_nsamples<<std::endl;
	m_dim += (int) shape[1];

	std::cout << "shape: ";
	for (size_t i = 0; i<shape.size(); i++)
		std::cout << shape[i] << ", ";
	std::cout << std::endl;
	std::cout << "fortran order: " << (fortran_order ? "+" : "-");
	std::cout << std::endl;

	for(auto i = 0; i<std::min(m_nsamples, maxiter); i++)
	{	
		if(i % 10000 == 0)
			std::cout<<"Reading..."<<i<<std::endl;
		for(int j = 0; j<m_dim; j++)
		{
			int index = i*m_dim + j;
			m_instances[i].push_back((double) data[index]);
		}
	}
	// Read true_labels file

	// m_dim ++;

	// std::ifstream fin(true_labels);
	
	// if (fin.fail()) {
	// 	std::cout<<"Cannot read from file "<<file<<" !"<<std::endl;
	// 	exit(1);
	// }

    // std::string line, word, temp;
	// int it = 0;

	// while (getline(fin, line)) {
	// 	std::stringstream s(line);
    //     while (getline(s, word, ',')) { 
    //         // add all the column data 
    //         // of a row to a vector 
    //         const char* val = word.c_str();
    //         m_instances[it].push_back(categories[val]);
    //     }
	// 	it++;
	// }
	// std::cout<<it<<std::endl;
	// assert(it == m_nsamples);

	// fin.close();
}

// Dataset::Dataset(const char* file) {
// 	m_nsamples = 0;
// 	m_dim = -1;

// 	std::ifstream fin(file);
	
// 	if (fin.fail()) {
// 		std::cout<<"Cannot read from file "<<file<<" !"<<std::endl;
// 		exit(1);
// 	}
	
// 	std::vector<double> row; 
//     std::string line, word, temp; 

// 	while (getline(fin, line)) {
// 		row.clear();
//         std::stringstream s(line);
//         int ncols = 0;
//         while (getline(s, word, ',')) { 
//             // add all the column data 
//             // of a row to a vector 
//             double val = std::atof(word.c_str());
//             row.push_back(val);
//             ncols++;
//         }
//         m_instances.push_back(row);
//         if (m_dim==-1) m_dim = ncols;
//         else if (m_dim!=ncols) {
//         	std::cerr << "ERROR, inconsistent dataset" << std::endl;
//         	exit(-1);
//         }
// 		m_nsamples ++;
// 	}
// 	fin.close();
// }

Dataset::Dataset(const std::vector<std::vector<double> > &vector_of_vector) {
	m_instances = vector_of_vector;
	m_dim = vector_of_vector[0].size();
	m_nsamples = vector_of_vector.size();
}

std::vector<double> Dataset::getInstance(int i) {

    // std::cout<<i<<std::endl;
	return m_instances[i];
}
