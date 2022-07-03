#include "factors/MarginalizationFactor.h"

void *ThreadsConstructA(void *threadsstruct) {
    ThreadsStruct *p = ((ThreadsStruct *) threadsstruct);
    // 遍历分配给该线程的因子
    for (auto it : p->sub_factors) {
        // 遍历参数块
        for (int i = 0; i < static_cast<int>(it->parameter_blocks.size()); i++) {
            // 取第i个参数在H矩阵中的索引
            int idx_i = p->parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[i])];
            // 取第i个参数变量维度
            int size_i = p->parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[i])];
            if (size_i == 4)
                size_i = 3;
            // jacobian_i : 该因子关于第i个参数的雅克比矩阵 右边size_i列 (该参数变量有多少维就取右边多少列)
            Eigen::MatrixXd jacobian_i = it->jacobians[i].rightCols(size_i);
            /***************************************
            ** qpc Debug:
            ****************************************/
            //std::cout<<" it->jacobians[i] "<<it->jacobians[i].rows()<<" "<<it->jacobians[i].cols()<<std::endl;
            //std::cout<<" jacobian_i "<<jacobian_i.rows()<<" "<<jacobian_i.cols()<<std::endl;
            /***************************************
            ** Debug Finish!
            ** 结论：
            **  ".rightCols(size_i);"操作是对于四元数
            **  而言的，因为其it->jacobians[i]是1x4矩阵，
            **  但是实际上只需要右边的1x3矩阵。
            ****************************************/
            // 从对角线开始遍历
            for (int j = i; j < static_cast<int>(it->parameter_blocks.size()); j++) {
                int idx_j = p->parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[j])];
                int size_j = p->parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[j])];
                if (size_j == 4)
                    size_j = 3;
                Eigen::MatrixXd jacobian_j = it->jacobians[j].rightCols(size_j);
                if (i == j)     // 对角线
                    p->A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                else {  //非对角线
                    p->A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                    p->A.block(idx_j, idx_i, size_j, size_i) = p->A.block(idx_i, idx_j, size_i, size_j).transpose();
                }
            }
            // J^T J dx = J^T b
            p->b.segment(idx_i, size_i) += jacobian_i.transpose() * it->residuals;
        }
    }
    return threadsstruct;
}

void ResidualBlockInfo::Evaluate() {
    residuals.resize(cost_function->num_residuals());
    // 取各个参数块的size
    std::vector<int> block_sizes = cost_function->parameter_block_sizes();
    // double * [4] :一个四个元素的指针数组，每个元素可指向一个double的数据
    // double ** p=new double* [4]:定义一个p指针变量，指向指针数组的首元素
    // 有多少个参数块，就new 一个有多少个元素的指针数组
    raw_jacobians = new double *[block_sizes.size()];
    // 有多少个参数块，就resize 多少个Eigen矩阵
    jacobians.resize(block_sizes.size());

    for (int i = 0; i < static_cast<int>(block_sizes.size()); i++) {
        // 第i个参数块的雅克比矩阵 , resize (残差维度，该参数块维度)
        jacobians[i].resize(cost_function->num_residuals(), block_sizes[i]);
        // 指针指向 第i个参数块的雅克比矩阵
        raw_jacobians[i] = jacobians[i].data();
    }

    // 调用 cost_function 计算残差、雅克比
    cost_function->Evaluate(parameter_blocks.data(), residuals.data(), raw_jacobians);

    // 如果有用到核函数
    if (loss_function) {
        double residual_scaling_, alpha_sq_norm_;

        double sq_norm, rho[3];

        sq_norm = residuals.squaredNorm();
        loss_function->Evaluate(sq_norm, rho);

        double sqrt_rho1_ = sqrt(rho[1]);

        if ((sq_norm == 0.0) || (rho[2] <= 0.0)) {
            residual_scaling_ = sqrt_rho1_;
            alpha_sq_norm_ = 0.0;
        } else {
            const double D = 1.0 + 2.0 * sq_norm * rho[2] / rho[1];
            const double alpha = 1.0 - sqrt(D);
            residual_scaling_ = sqrt_rho1_ / (1 - alpha);
            alpha_sq_norm_ = alpha / sq_norm;
        }

        for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++) {
            jacobians[i] = sqrt_rho1_ * (jacobians[i] - alpha_sq_norm_ * residuals * (residuals.transpose() * jacobians[i]));

        }

        residuals *= residual_scaling_;
    }
}

MarginalizationInfo::~MarginalizationInfo() {
    for (auto it = parameter_block_data.begin(); it != parameter_block_data.end(); ++it)
        delete it->second;

    for (int i = 0; i < (int) factors.size(); i++) {

        delete[] factors[i]->raw_jacobians;

        delete factors[i]->cost_function;

        delete factors[i];
    }
}

void MarginalizationInfo::AddResidualBlockInfo(ResidualBlockInfo *residual_block_info) {
    factors.emplace_back(residual_block_info);

    /// t: 平移量参数块
    /// q: 旋转量参数块
    /// M: speedbias参数块
    /// IMU因子: [t0,q0,M0,t1,q1,M1]
    /// speedBias先验: [Mi] (i=1,2,...width-1) (就是添加了多个)
    /// Lidar边缘线因子: [t0,q0]
    /// Lidar平面因子: [t0,q0]
    std::vector<double *> &parameter_blocks = residual_block_info->parameter_blocks;

    /// IMU因子: [3,4,9,3,4,9]
    /// speedBias先验: [9]
    /// Lidar边缘线因子: [3,4]
    /// Lidar平面因子: [3,4]
    std::vector<int> parameter_block_sizes = residual_block_info->cost_function->parameter_block_sizes();

    // 遍历参数块
    for (int i = 0; i < static_cast<int>(residual_block_info->parameter_blocks.size()); i++) {
        // 取参数块地址
        double *addr = parameter_blocks[i];
        // 取参数块size
        int size = parameter_block_sizes[i];
        // 做映射
        parameter_block_size[reinterpret_cast<long>(addr)] = size;
    }

    // 如果没有要marg的变量，直接返回
    if(residual_block_info->drop_set.size() == 0) return;

    // 遍历要marg的变量
    for (int i = 0; i < static_cast<int>(residual_block_info->drop_set.size()); i++) {
        // 取要marg的参数块地址
        double *addr = parameter_blocks[residual_block_info->drop_set[i]];
        // 做映射, 首先初始化为0
        parameter_block_idx[reinterpret_cast<long>(addr)] = 0;
    }
}

// 计算每个因子对应的
// 1. 误差项(residuals)
// 2. 雅可比矩阵(jacobians)
// 并将参数变量放到parameter_block_data
void MarginalizationInfo::PreMarginalize() {
    // 遍历因子
    for (auto it : factors) {
        //
        it->Evaluate();

        // 取各个参数块的size
        std::vector<int> block_sizes = it->cost_function->parameter_block_sizes();
        for (int i = 0; i < static_cast<int>(block_sizes.size()); i++) {
            // 取参数块地址
            long addr = reinterpret_cast<long>(it->parameter_blocks[i]);
            // 取该参数块size
            int size = block_sizes[i];
            // 如果parameter_block_data还没有该地址的记录
            if (parameter_block_data.find(addr) == parameter_block_data.end()) {
                // new 一个数组
                double *data = new double[size];
                // 将该参数块的数据复制到上述数组
                memcpy(data, it->parameter_blocks[i], sizeof(double) * size);
                // 做映射
                parameter_block_data[addr] = data;
            }
        }
    }
}

int MarginalizationInfo::LocalSize(int size) const {
    // 如果是四元数，实际参数只有3个
    return size == 4 ? 3 : size;
}

void MarginalizationInfo::Marginalize() {
    int pos = 0;
    // 遍历要marg的变量
    for (auto &it : parameter_block_idx) {
        // 在这里设置该变量在H矩阵中的索引
        it.second = pos;
        pos += LocalSize(parameter_block_size[it.first]);
    }

    // m: 需要marg掉的变量的总维度
    m = pos;

    // 遍历所有变量
    for (const auto &it : parameter_block_size) {
        // 现在添加要保留的变量
        if (parameter_block_idx.find(it.first) == parameter_block_idx.end()) {
            parameter_block_idx[it.first] = pos;
            pos += LocalSize(it.second);
        }
    }
    // n: 需要保留的变量的总维度
    n = pos - m;

    // pos: 所有与要marg的变量相关的变量总维度
    Eigen::MatrixXd A(pos, pos);
    Eigen::VectorXd b(pos);
    A.setZero();
    b.setZero();

    // 线程数组
    pthread_t tids[NUM_THREADS];
    // 要传入线程的参数
    ThreadsStruct threadsstruct[NUM_THREADS];
    // 遍历因子
    int i = 0;
    for (auto it : factors) {
        // 给线程分配因子
        threadsstruct[i].sub_factors.push_back(it);
        i++;
        i = i % NUM_THREADS;
    }
    for (int i = 0; i < NUM_THREADS; i++) {
        // 初始化
        threadsstruct[i].A = Eigen::MatrixXd::Zero(pos, pos);
        threadsstruct[i].b = Eigen::VectorXd::Zero(pos);
        threadsstruct[i].parameter_block_size = parameter_block_size;
        threadsstruct[i].parameter_block_idx = parameter_block_idx;
        // 启动各个线程
        int ret = pthread_create(&tids[i], NULL, ThreadsConstructA, (void *) &(threadsstruct[i]));
        if (ret != 0) {
            ROS_DEBUG("pthread_create error");
            ROS_BREAK();
        }
    }
    // 线程结果累加
    for (int i = NUM_THREADS - 1; i >= 0; i--) {
        pthread_join(tids[i], NULL);
        A += threadsstruct[i].A;
        b += threadsstruct[i].b;
    }

    ///          m              r
    ///   |--------------|--------------|
    ///   |              |              |
    /// m |     Amm      |     Amr      |
    ///   |              |              |
    ///   |--------------|--------------|
    ///   |              |              |
    /// r |     Arm      |     Arr      |
    ///   |              |              |
    ///   |--------------|--------------|

    // H矩阵左上角(mxm)，是要被marg的部分
    // 确保是对称的
    Eigen::MatrixXd Amm = 0.5 * (A.block(0, 0, m, m) + A.block(0, 0, m, m).transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Amm);

    // 求逆
    Eigen::MatrixXd Amm_inv = saes.eigenvectors()
            * Eigen::VectorXd((saes.eigenvalues().array() > eps).select(saes.eigenvalues().array().inverse(), 0)).asDiagonal()
            * saes.eigenvectors().transpose();
    // bmm: 要marg的残差
    Eigen::VectorXd bmm = b.segment(0, m);
    //
    Eigen::MatrixXd Amr = A.block(0, m, m, n);
    Eigen::MatrixXd Arm = A.block(m, 0, n, m);
    Eigen::MatrixXd Arr = A.block(m, m, n, n);
    Eigen::VectorXd brr = b.segment(m, n);
    // 直接Schur补
    A = Arr - Arm * Amm_inv * Amr;
    b = brr - Arm * Amm_inv * bmm;

    // 从边缘化后得到的A和b中恢复出线性化的雅克比和残差
    //这步操作的目的为：恢复出雅克比和残差之后，便可以重新构造成factor和其他factor一起通过ceres进行非线性优化迭代
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(A);
    Eigen::VectorXd S = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array(), 0));
    Eigen::VectorXd
            S_inv = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array().inverse(), 0));

    Eigen::VectorXd S_sqrt = S.cwiseSqrt();
    Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();

    linearized_jacobians = S_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
    linearized_residuals = S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose() * b;
}

std::vector<double *> MarginalizationInfo::GetParameterBlocks(std::unordered_map<long, double *> &addr_shift) {
    std::vector<double *> keep_block_addr;
    keep_block_size.clear();
    keep_block_idx.clear();
    keep_block_data.clear();
    // 遍历marg因子的变量
    for (const auto &it : parameter_block_idx) {
        // >=m 表示是需要保留的变量
        if (it.second >= m) {
            // 要保留的变量维度
            keep_block_size.push_back(parameter_block_size[it.first]);
            // 要保留的变量在H矩阵的索引
            keep_block_idx.push_back(parameter_block_idx[it.first]);
            // 要保留的变量数据
            keep_block_data.push_back(parameter_block_data[it.first]);
            // 要保留的变量的在下一个H矩阵构造时的地址
            keep_block_addr.push_back(addr_shift[it.first]);
        }
    }
    sum_block_size = std::accumulate(std::begin(keep_block_size), std::end(keep_block_size), 0);

    return keep_block_addr;
}

MarginalizationFactor::MarginalizationFactor(MarginalizationInfo *_marginalization_info) : marginalization_info(
                                                                                               _marginalization_info) {
    int cnt = 0;
    for (auto it : marginalization_info->keep_block_size) {
        mutable_parameter_block_sizes()->push_back(it);
        cnt += it;
    }
    set_num_residuals(marginalization_info->n);
};

bool MarginalizationFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
    int n = marginalization_info->n;
    int m = marginalization_info->m;
    Eigen::VectorXd dx(n);
    for (int i = 0; i < static_cast<int>(marginalization_info->keep_block_size.size()); i++) {
        int size = marginalization_info->keep_block_size[i];
        int idx = marginalization_info->keep_block_idx[i] - m;
        Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(parameters[i], size);
        Eigen::VectorXd x0 = Eigen::Map<const Eigen::VectorXd>(marginalization_info->keep_block_data[i], size);
        if (size != 4) {
            dx.segment(idx, size) = x - x0;
        }
        else {
            dx.segment<3>(idx) = 2.0 * (Eigen::Quaterniond(x0(0), x0(1), x0(2), x0(3)).inverse()
                                        * Eigen::Quaterniond(x(0), x(1), x(2), x(3))).normalized().vec();
            if ((Eigen::Quaterniond(x0(0), x0(1), x0(2), x0(3)).inverse() * Eigen::Quaterniond(x(0), x(1), x(2), x(3))).w()
                    < 0) {
                dx.segment<3>(idx) = -2.0 * (Eigen::Quaterniond(x0(0), x0(1), x0(2), x0(3)).inverse()
                                             * Eigen::Quaterniond(x(0), x(1), x(2), x(3))).normalized().vec();
            }
        }
    }

    Eigen::Map<Eigen::VectorXd>(residuals, n) =
            marginalization_info->linearized_residuals + marginalization_info->linearized_jacobians * dx;

    if (jacobians) {

        for (int i = 0; i < static_cast<int>(marginalization_info->keep_block_size.size()); i++) {
            if (jacobians[i]) {

                int size = marginalization_info->keep_block_size[i], local_size = marginalization_info->LocalSize(size);
                int idx = marginalization_info->keep_block_idx[i] - m;
                //
                Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(parameters[i], size);
                Eigen::VectorXd x0 = Eigen::Map<const Eigen::VectorXd>(marginalization_info->keep_block_data[i], size);
                //
                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                        jacobian(jacobians[i], n, size);
                jacobian.setZero();
                if(size != 4)
                    jacobian.rightCols(local_size) = marginalization_info->linearized_jacobians.middleCols(idx, local_size);
                else {
                    if ((Eigen::Quaterniond(x0(0), x0(1), x0(2), x0(3)).inverse() * Eigen::Quaterniond(x(0), x(1), x(2), x(3))).w() >= 0)
                        jacobian.rightCols(size) = 2.0 * marginalization_info->linearized_jacobians.middleCols(idx, local_size) *
                                Qleft(Eigen::Quaterniond(x0(0), x0(1), x0(2), x0(3)).inverse()).bottomRightCorner<3, 4>();
                    else
                        jacobian.rightCols(size) = -2.0 * marginalization_info->linearized_jacobians.middleCols(idx, local_size) *
                                Qleft(Eigen::Quaterniond(x0(0), x0(1), x0(2), x0(3)).inverse()).bottomRightCorner<3, 4>();
                }
            }
        }
    }
    return true;
}
