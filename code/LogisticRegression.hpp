#include <Eigen/Dense>
#include <Eigen/Core>
#include "Dataset.hpp"
#include "Classification.hpp"
#include <set>

#ifndef LOGISTICREGRESSION_HPP
#define LOGISTICREGRESSION_HPP
/**
  The LogisticRegression class inherits from the Regression class, stores the coefficient and provides a bunch of specific methods.
*/
class LogisticRegression : public Classification {
private:
    /**
      The classes.
    */
  std::set<double> m_class_type;
    /**
      The number of classes.
    */
	int m_class_no;

	Eigen::VectorXd* m_beta1;
  Eigen::MatrixXd* m_X;
  Eigen::MatrixXd* m_Z;

public:
    /**
      The linear regression method fits a linear regression coefficient to col_regr using the provided Dataset. It calls setCoefficients under the hood.
     @param dataset a pointer to a dataset
     @param m_col_class the integer of the column index of Y
    */
	LogisticRegression(Dataset* dataset, int col_class);
    /**
      The destructor (frees m_beta).
    */
  ~LogisticRegression();
    /**
      The setter method of the private attribute m_beta which is called by LogisticRegression.
    */
	void Setup();
  double sigmoidfcn(double t);
  void gradient(Eigen::VectorXd* res, double lambda);
  void sec_gradient(Eigen::MatrixXd* res, double lambda);
  void newton_raphson(double lambda, double conv_thres=1);
    /**
      The estimate method outputs the predicted Y for a given point x.
     @param x the point for which to estimate Y.
    */
	int Estimate( const Eigen::VectorXd & x , double threshold=0.5);

  int getClassNo();
};

#endif //LOGISTICREGRESSION_HPP
