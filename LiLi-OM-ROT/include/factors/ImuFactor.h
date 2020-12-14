#pragma once
#include <ros/assert.h>
#include <iostream>
#include <eigen3/Eigen/Dense>

#include "utils/math_tools.h"
#include "Preintegration.h"

#include <ceres/ceres.h>

//ceres::SizedCostFunction<15, 3, 4, 9, 3, 4, 9>
//<误差维度，p1维度，q1维度，speed1&bias1维度 , p2维度，q2维度，speed2&bias2维度>
class ImuFactor : public ceres::SizedCostFunction<15, 3, 4, 9, 3, 4, 9> {

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    ImuFactor() = delete;
    ImuFactor(Preintegration *pre_integration) : pre_integration_{
                                                     pre_integration} {
        g_vec_ = pre_integration_->g_vec_;
    }

    // 调用 cost_function 计算残差、雅克比
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {

        /// 待优化参数块1
        Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond Qi(parameters[1][0], parameters[1][1], parameters[1][2], parameters[1][3]);
        Qi.normalize();

        Eigen::Vector3d Vi(parameters[2][0], parameters[2][1], parameters[2][2]);
        Eigen::Vector3d Bai(parameters[2][3], parameters[2][4], parameters[2][5]);
        Eigen::Vector3d Bgi(parameters[2][6], parameters[2][7], parameters[2][8]);

        /// 待优化参数块2
        Eigen::Vector3d Pj(parameters[3][0], parameters[3][1], parameters[3][2]);
        Eigen::Quaterniond Qj(parameters[4][0], parameters[4][1], parameters[4][2], parameters[4][3]);
        Qj.normalize();

        Eigen::Vector3d Vj(parameters[5][0], parameters[5][1], parameters[5][2]);
        Eigen::Vector3d Baj(parameters[5][3], parameters[5][4], parameters[5][5]);
        Eigen::Vector3d Bgj(parameters[5][6], parameters[5][7], parameters[5][8]);

        /// 计算残差
        Eigen::Map<Eigen::Matrix<double, 15, 1>> residual(residuals);
        residual = pre_integration_->evaluate(Pi, Qi, Vi, Bai, Bgi,
                                              Pj, Qj, Vj, Baj, Bgj);

        /// 取协方差
        Eigen::Matrix<double, 15, 15> sqrt_info =
                Eigen::LLT<Eigen::Matrix<double, 15, 15>>(pre_integration_->covariance_.inverse()).matrixL().transpose();

        ///这里残差 residual 乘以 sqrt_info，这是因为真正的优化项其实是 Mahalanobis 距离:d= [r^T] [P^{-1}] [r] , 其中 [P] 是协方差。
        /// Mahalanobis距离其实相当于一个残差加权，协方差大的加权小，协方差小的加权大，着重优化那些比较确定的残差
        residual = sqrt_info * residual;

        ///计算雅克比（可参考Vins-Mono）
        /// 预积分误差对待求状态变量的偏导
        if (jacobians) {
            double sum_dt = pre_integration_->sum_dt_;
            Eigen::Matrix3d dp_dba = pre_integration_->jacobian_.template block<3, 3>(O_P, O_BA);
            Eigen::Matrix3d dp_dbg = pre_integration_->jacobian_.template block<3, 3>(O_P, O_BG);

            Eigen::Matrix3d dq_dbg = pre_integration_->jacobian_.template block<3, 3>(O_R, O_BG);

            Eigen::Matrix3d dv_dba = pre_integration_->jacobian_.template block<3, 3>(O_V, O_BA);
            Eigen::Matrix3d dv_dbg = pre_integration_->jacobian_.template block<3, 3>(O_V, O_BG);

            if (pre_integration_->jacobian_.maxCoeff() > 1e8 || pre_integration_->jacobian_.minCoeff() < -1e8) {
                ROS_WARN("numerical unstable in preintegration");
            }

            ///预积分误差对 Pi的雅克比
            if (jacobians[0]) {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
                jacobian_pose_i.setZero();

                // 预积分位置误差对 Pi的雅克比
                jacobian_pose_i.block<3, 3>(O_P, O_P) = -Qi.inverse().toRotationMatrix();

                jacobian_pose_i = sqrt_info * jacobian_pose_i;

                if (jacobian_pose_i.maxCoeff() > 1e8 || jacobian_pose_i.minCoeff() < -1e8) {
                    ROS_WARN("numerical unstable in preintegration");
                }
            }


            ///预积分误差对 Qi的雅克比
            if (jacobians[1]) {
                Eigen::Map<Eigen::Matrix<double, 15, 4, Eigen::RowMajor>> jacobian_pose_qi(jacobians[1]);
                jacobian_pose_qi.setZero();

                Eigen::Vector3d tmp = -0.5 * g_vec_ * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt;

                // 预积分位置误差 对Qi的雅克比
                jacobian_pose_qi.block<3, 1>(O_P, 0) = 2 * (Qi.w() * tmp + skewSymmetric(Qi.vec()) * tmp);
                jacobian_pose_qi.block<3, 3>(O_P, 1) = 2 * (Qi.vec().dot(tmp) * Eigen::Matrix3d::Identity() + Qi.vec() * tmp.transpose() - tmp * Qi.vec().transpose() - Qi.w() * skewSymmetric(tmp));

                // 预积分姿态误差 对Qi的雅克比
                Eigen::Quaterniond corrected_delta_q =
                        pre_integration_->delta_q_ * deltaQ(dq_dbg * (Bgi - pre_integration_->linearized_bg_));
                jacobian_pose_qi.block<3, 4>(O_R, 0) =
                        -2 * (Qleft(Qj.inverse()) * Qright(corrected_delta_q)).bottomRightCorner<3, 4>();


                // 预积分速度误差 对Qi的雅克比
                Eigen::Vector3d tmp1 = -g_vec_ * sum_dt + Vj - Vi;

                jacobian_pose_qi.block<3, 1>(O_V, 0) = 2 * (Qi.w() * tmp1 + skewSymmetric(Qi.vec()) * tmp1);
                jacobian_pose_qi.block<3, 3>(O_V, 1) = 2 * (Qi.vec().dot(tmp1) * Eigen::Matrix3d::Identity() + Qi.vec() * tmp1.transpose() - tmp1 * Qi.vec().transpose() - Qi.w() * skewSymmetric(tmp1));

                jacobian_pose_qi = sqrt_info * jacobian_pose_qi;
            }


            ///预积分误差对 Vi,Bias_i的雅克比
            if (jacobians[2]) {
                Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_speedbias_i(jacobians[2]);
                jacobian_speedbias_i.setZero();
                // 预积分位置误差 对Vi,Bias_i的雅克比
                jacobian_speedbias_i.block<3, 3>(O_P, O_V - O_V) = -Qi.inverse().toRotationMatrix() * sum_dt;
                jacobian_speedbias_i.block<3, 3>(O_P, O_BA - O_V) = -dp_dba;
                jacobian_speedbias_i.block<3, 3>(O_P, O_BG - O_V) = -dp_dbg;

                // 预积分姿态误差 对Vi,Bias_i的雅克比
                Eigen::Quaterniond corrected_delta_q =
                        pre_integration_->delta_q_ * deltaQ(dq_dbg * (Bgi - pre_integration_->linearized_bg_));
                jacobian_speedbias_i.block<3, 3>(O_R, O_BG - O_V) =
                        -LeftQuatMatrix(Qj.inverse() * Qi * corrected_delta_q).topLeftCorner<3, 3>() * dq_dbg;

                // 预积分速度误差 对Vi,Bias_i的雅克比
                jacobian_speedbias_i.block<3, 3>(O_V, O_V - O_V) = -Qi.inverse().toRotationMatrix();
                jacobian_speedbias_i.block<3, 3>(O_V, O_BA - O_V) = -dv_dba;
                jacobian_speedbias_i.block<3, 3>(O_V, O_BG - O_V) = -dv_dbg;

                // 预积分ba，bg误差 对Vi,Bias_i的雅克比
                jacobian_speedbias_i.block<3, 3>(O_BA, O_BA - O_V) = -Eigen::Matrix3d::Identity();

                jacobian_speedbias_i.block<3, 3>(O_BG, O_BG - O_V) = -Eigen::Matrix3d::Identity();

                jacobian_speedbias_i = sqrt_info * jacobian_speedbias_i;

            }

            ///预积分误差对 Pj的雅克比
            if (jacobians[3]) {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> jacobian_pose_j(jacobians[3]);
                jacobian_pose_j.setZero();

                //预积分位置误差对 Pj的雅克比
                jacobian_pose_j.block<3, 3>(O_P, O_P) = Qi.inverse().toRotationMatrix();

                jacobian_pose_j = sqrt_info * jacobian_pose_j;

            }

            ///预积分误差对 Qj的雅克比
            if (jacobians[4]) {
                Eigen::Map<Eigen::Matrix<double, 15, 4, Eigen::RowMajor>> jacobian_pose_qj(jacobians[4]);
                jacobian_pose_qj.setZero();

                //预积分姿态误差对 Qj的雅克比
                Eigen::Quaterniond corrected_delta_q =
                        pre_integration_->delta_q_ * deltaQ(dq_dbg * (Bgi - pre_integration_->linearized_bg_));
                jacobian_pose_qj.block<3, 4>(O_R, 0) =
                        2 * Qleft(corrected_delta_q.inverse() * Qi.inverse()).bottomRightCorner<3, 4>();

                jacobian_pose_qj = sqrt_info * jacobian_pose_qj;

            }

            ///预积分误差对 Vj,Bias_j的雅克比
            if (jacobians[5]) {
                Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_speedbias_j(jacobians[5]);
                jacobian_speedbias_j.setZero();

                // 预积分速度误差 对Vj,Bias_j的雅克比
                jacobian_speedbias_j.block<3, 3>(O_V, O_V - O_V) = Qi.inverse().toRotationMatrix();

                // 预积分ba误差 对Vj,Bias_j的雅克比
                jacobian_speedbias_j.block<3, 3>(O_BA, O_BA - O_V) = Eigen::Matrix3d::Identity();
                // 预积分bg误差 对Vj,Bias_j的雅克比
                jacobian_speedbias_j.block<3, 3>(O_BG, O_BG - O_V) = Eigen::Matrix3d::Identity();

                jacobian_speedbias_j = sqrt_info * jacobian_speedbias_j;

            }
        }

        return true;
    }

    Preintegration *pre_integration_;
    Eigen::Vector3d g_vec_;
};
