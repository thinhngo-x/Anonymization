#include "LinearRegression.hpp"
#include "Dataset.hpp"
#include "Classification.hpp"
#include<iostream>
#include<cassert>
#include <set>
#include <cfloat>


LinearRegression::LinearRegression( Dataset* dataset, int col_class) 
: Classification(dataset, col_class) {
	//find number of class at m_col_class
	for (int i = 0; i < m_dataset->getNbrSamples(); i++){
		std::vector<double> temp = m_dataset->getInstance(i);
		m_class_type.insert(temp[m_col_class]);
    }
	m_class_no = m_class_type.size();
	SetCoefficients();
}

LinearRegression::~LinearRegression() {
	delete m_beta;
}

void LinearRegression::SetCoefficients() {
	int n = m_dataset->getNbrSamples();
	int d = m_dataset->getDim()-1;
	Eigen::MatrixXd X = Eigen::MatrixXd(n,d+1);
	Eigen::MatrixXd Z = Eigen::MatrixXd(n,m_class_no);
	m_beta = new Eigen::MatrixXd(d+1,m_class_no);
	m_beta->setZero(d+1,m_class_no);
	for (int i = 0; i < n; i++){
		std::vector<double> temp = m_dataset->getInstance(i);
		for (int j = 0; j < d+1; j++){
			if (j == 0){
				X(i,j) = 1;
			}
			else if (j <= m_col_class){
				X(i,j) = temp[j-1];
			}
			else if (j > m_col_class){
				X(i,j) = temp[j];
			}
		}
		std::set<double>::iterator it = m_class_type.begin();
        for (int j = 0; j < m_class_no; j++){
            Z(i,j) = (temp[m_col_class] == *it)? 1 : 0;
			it++;
        }
	}
	Eigen::MatrixXd res = (X.transpose()*X).inverse()*X.transpose()*Z;
	//std::cout<< Z <<std::endl;
	*m_beta += res;
}

void LinearRegression::ShowCoefficients() {
	if (!m_beta) {
		std::cout<<"Coefficients have not been allocated."<<std::endl;
		return;
	}
	
	if (m_beta->size() != m_dataset->getDim()*m_class_no) {
		std::cout<< "Warning, unexpected size of coefficients matrix: " << m_beta->size() << std::endl;
	}
	
	std::cout<< "beta = (";
	for (int i=0; i<m_dataset->getDim(); i++) {
        for (int j=0; j<m_class_no; j++) {
		    std::cout<< " " << (*m_beta)(i,j);
        }
	}
	std::cout << " )"<<std::endl;
}

int LinearRegression::Estimate( const Eigen::VectorXd & x ,double threshold) { //threshold not used here, just for virtual fcn
	int d = m_dataset->getDim()-1;
	Eigen::VectorXd vec = Eigen::VectorXd(d+1);
	for (int i = 0; i < d+1; i++){
		if (i == 0){
			vec(i) = 1;
		}
		else{
			vec(i) = x(i-1);
		}
	}
	Eigen::VectorXd delta = vec.transpose()*(*m_beta);
    double max = DBL_MIN;
    int index = -1;
    for (int i = 0; i < m_class_no; i++){
        if (delta[i] > max){
            max = delta[i];
            index = i;
        }
    }
    return index;
}

int LinearRegression::getClassNo(){
	return m_class_no;
}
