//binary

#include "LogisticRegression.hpp"
#include "Dataset.hpp"
#include "Classification.hpp"
#include<iostream>
#include<cassert>
#include <set>
#include <cfloat>
#include <cmath>


LogisticRegression::LogisticRegression( Dataset* dataset, int col_class) 
: Classification(dataset, col_class) {
	//find number of class at m_col_class
	for (int i = 0; i < m_dataset->getNbrSamples(); i++){
		std::vector<double> temp = m_dataset->getInstance(i);
		m_class_type.insert(temp[m_col_class]);
    }
	m_class_no = m_class_type.size();
	assert(m_class_no == 2); //here is binary, thus 2
	Setup();
}

LogisticRegression::~LogisticRegression() {
	delete m_beta1;
	delete m_X;
	delete m_Z;
}

void LogisticRegression::Setup() {
	int n = m_dataset->getNbrSamples();
	int d = m_dataset->getDim()-1;
	Eigen::MatrixXd X = Eigen::MatrixXd(n,d+1);
	Eigen::MatrixXd Z = Eigen::MatrixXd(n,m_class_no);

	m_X = new Eigen::MatrixXd(n,d+1);
	m_X->setZero(n,d+1);
	m_Z = new Eigen::MatrixXd(n,m_class_no);
	m_Z->setZero(n,m_class_no);

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
	*m_X += X;
	*m_Z += Z;
	double lambda = 5;
	newton_raphson(lambda,1);
}

double LogisticRegression::sigmoidfcn(double t){
	return 1/(1+exp(-t));
}

void LogisticRegression::gradient(Eigen::VectorXd* res, double lambda){
	for (int i = 0; i < m_dataset->getNbrSamples(); i++){
		Eigen::VectorXd temp = m_X->row(i);
		*res += ((*m_Z)(i,0) - sigmoidfcn(temp.transpose()*(*m_beta1)))*temp.transpose(); //z_i to check whether in first class
	}
	*res -= 2*lambda*(*m_beta1);
}

void LogisticRegression::sec_gradient(Eigen::MatrixXd* res, double lambda){
	int d = m_dataset->getDim()-1;
	for (int i = 0; i < m_dataset->getNbrSamples(); i++){
		Eigen::VectorXd temp = m_X->row(i);
		*res -= sigmoidfcn(temp.transpose()*(*m_beta1))*(1-sigmoidfcn(temp.transpose()*(*m_beta1)))*temp*temp.transpose();
	}
	Eigen::MatrixXd iden = Eigen::MatrixXd(d+1,d+1);
	iden.setIdentity();
	*res -= 2*lambda*iden;
}

void LogisticRegression::newton_raphson(double lambda, double conv_thres){
	int d = m_dataset->getDim()-1;
	m_beta1 = new Eigen::VectorXd(d+1);
	m_beta1->setZero();
	Eigen::VectorXd res = Eigen::VectorXd(d+1);
	res.setZero();
	double dist = 0.0;
	do {
		Eigen::VectorXd grad = Eigen::VectorXd(d+1);
		Eigen::MatrixXd grad2 = Eigen::MatrixXd(d+1,d+1);
		gradient(&grad,lambda);
		sec_gradient(&grad2,lambda);
		res = *m_beta1 - grad2.inverse()*grad;
		dist = 0;
		for (int i = 0; i < d+1; i++){
			dist += (res(i)-(*m_beta1)(i))*(res(i)-(*m_beta1)(i));
		}
		dist = std::sqrt(dist);
		*m_beta1 = res;
	} while (dist > conv_thres);
}

int LogisticRegression::Estimate( const Eigen::VectorXd & x ,double threshold) { //threshold not used here, just for virtual fcn
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
	Eigen::VectorXd delta(m_class_no);
	double delta_1 = sigmoidfcn(vec.transpose()*(*m_beta1));
	double delta_2 = 1 - delta_1;
	delta << delta_1, delta_2;

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

int LogisticRegression::getClassNo(){
	return m_class_no;
}
