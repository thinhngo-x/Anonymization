
#include "KnnClassification.hpp"
#include <iostream>
// #include <fstream>
#include <ANN/ANN.h>

 
KnnClassification::KnnClassification(int k, Dataset* dataset, int col_class)
: Classification(dataset, col_class) {
    ANNpointArray pa;

    // std::cout<<dataset->getDim()<<std::endl;
    pa = annAllocPts(dataset->getNbrSamples(), dataset->getDim()-1);

    // std::cout<<"YES"<<std::endl;
    for(int i = 0; i < dataset->getNbrSamples(); i++)
    {
        for(int j = 0; j < dataset->getDim(); j++)
            if (j < col_class)
                pa[i][j] = dataset->getInstance(i)[j];
            else if(j > col_class)
                pa[i][j-1] = dataset->getInstance(i)[j];
        

        // for(int j = 0; j < dataset->getDim()-1; j++)
        //     if(i == 0)
        //         std::cout<<pa[i][j]<<std::endl;
    }

    // std::cout<<<<std::endl;
    // std::cout<<"YES"<<std::endl;

    m_kdTree = new ANNkd_tree(pa, dataset->getNbrSamples(), dataset->getDim()-1);
    m_k = k;

    // m_kdTree->getStats();
    // std::cout<<"YES"<<std::endl;
}

KnnClassification::~KnnClassification() {
    // m_k = 0;
    annClose();
}

int KnnClassification::Estimate(const Eigen::VectorXd & x, double threshold) {
    int d = m_kdTree->theDim();
    ANNpoint px = annAllocPt(d);

    assert(d == x.size());
    for(int i = 0; i < d; i++)
    {
        px[i] = x(i);
        // std::cout<<px[i]<<std::endl;
    }
    ANNidxArray nn_idx = new ANNidx[m_k];
    ANNdistArray dist = new ANNdist[m_k];

    // std::cout<<m_k<<" "<<x<<" "<<dist<<std::endl;

    m_kdTree->annkSearch(px, m_k, nn_idx, dist);

    // std::cout<<"YES"<<std::endl;

    double p1 = 0.0;
    for(int i = 0; i < m_k; i++)
    {
        if (m_dataset->getInstance(nn_idx[i])[m_col_class] == 1)
            p1 += 1.0/m_k;
    }

    delete [] nn_idx;
    delete [] dist;

    if (p1 > threshold)
        return 1;
    else return 0;
}

int KnnClassification::getK() {
    return m_k;
}

ANNkd_tree* KnnClassification::getKdTree() {
    return m_kdTree;
}
