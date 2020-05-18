#include <Eigen/Dense>
#include <Eigen/Core>
#include "Dataset.hpp"
#include "Classification.hpp"
#include <set>

#ifndef LINEARREGRESSION_HPP
#define LINEARREGRESSION_HPP
/**
  The LinearRegression class inherits from the Regression class, stores the coefficient and provides a bunch of specific methods.
*/
class LinearRegression : public Classification {
private:
    /**
      The classes.
    */
  std::set<double> m_class_type;
    /**
      The number of classes.
    */
	int m_class_no;
    /**
      The linear regression coefficient.
    */
	Eigen::MatrixXd* m_beta;
public:
    /**
      The linear regression method fits a linear regression coefficient to col_regr using the provided Dataset. It calls setCoefficients under the hood.
     @param dataset a pointer to a dataset
     @param m_col_class the integer of the column index of Y
    */
	LinearRegression(Dataset* dataset, int col_class);
    /**
      The destructor (frees m_beta).
    */
  ~LinearRegression();
    /**
      The setter method of the private attribute m_beta which is called by LinearRegression.
    */
	void SetCoefficients();
    /**
      The getter method of the private attribute m_beta.
    */
	void ShowCoefficients();
    /**
      The estimate method outputs the predicted Y for a given point x.
     @param x the point for which to estimate Y.
    */
	int Estimate( const Eigen::VectorXd & x , double threshold=0.5);

  int getClassNo();
};

#endif //LINEARREGRESSION_HPP
