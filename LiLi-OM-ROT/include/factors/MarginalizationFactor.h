#ifndef MARGINALIZATIONFACTOR_H_
#define MARGINALIZATIONFACTOR_H_

#include <ceres/ceres.h>
#include <Eigen/Eigen>
#include <unordered_map>

#include "utils/common.h"
#include "utils/math_tools.h"

const int NUM_THREADS = 4;

struct ResidualBlockInfo {
    ResidualBlockInfo(ceres::CostFunction *_cost_function,
                      ceres::LossFunction *_loss_function,
                      std::vector<double *> _parameter_blocks,
                      std::vector<int> _drop_set)
        : cost_function(_cost_function),
          loss_function(_loss_function),
          parameter_blocks(_parameter_blocks),
          drop_set(_drop_set) {}

    void Evaluate();

    ceres::CostFunction *cost_function;
    ceres::LossFunction *loss_function;
    std::vector<double *> parameter_blocks;
    std::vector<int> drop_set;

    double **raw_jacobians;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;
    Eigen::VectorXd residuals;
};

class MarginalizationInfo {
public:
    ~MarginalizationInfo();
    int LocalSize(int size) const;
    void AddResidualBlockInfo(ResidualBlockInfo *residual_block_info);
    void PreMarginalize();
    void Marginalize();
    std::vector<double *> GetParameterBlocks(std::unordered_map<long, double *> &addr_shift);

    std::vector<ResidualBlockInfo *> factors;
    // m: 需要marg掉的变量的总维度
    // n: 需要保留的变量的总维度
    int m, n;
    std::unordered_map<long, int> parameter_block_size; //global size   映射每个变量的维度
    int sum_block_size;
    std::unordered_map<long, int> parameter_block_idx; //local size     映射每个变量在H矩阵中的索引,要marg的排在前面，后面是要保留的变量
    std::unordered_map<long, double *> parameter_block_data; // 变量数据

    std::vector<int> keep_block_size; //global size
    std::vector<int> keep_block_idx;  //local size
    std::vector<double *> keep_block_data;

    Eigen::MatrixXd linearized_jacobians;
    Eigen::VectorXd linearized_residuals;
    const double eps = 1e-8;
};

class MarginalizationFactor : public ceres::CostFunction {
public:
    MarginalizationFactor(MarginalizationInfo* _marginalization_info);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    MarginalizationInfo* marginalization_info;
};

struct ThreadsStruct {
    std::vector<ResidualBlockInfo *> sub_factors;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    std::unordered_map<long, int> parameter_block_size; //global size
    std::unordered_map<long, int> parameter_block_idx; //local size
};

#endif //MARGINALIZATIONFACTOR_H_
