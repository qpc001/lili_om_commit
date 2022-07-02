#include "utils/common.h"
#include "utils/math_tools.h"
#include "utils/timer.h"
#include "factors/LidarKeyframeFactor.h"
#include "factors/LidarPoseFactor.h"
#include "factors/ImuFactor.h"
#include "factors/PriorFactor.h"
#include "factors/Preintegration.h"
#include "factors/MarginalizationFactor.h"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/ISAM2.h>

using namespace gtsam;

class BackendFusion {
private:
    int odom_sub_cnt = 0;
    int map_pub_cnt = 0;
    ros::NodeHandle nh;

    ros::Subscriber sub_edge;
    ros::Subscriber sub_surf;
    ros::Subscriber sub_odom;
    ros::Subscriber sub_each_odom;
    ros::Subscriber sub_full_cloud;
    ros::Subscriber sub_imu;

    ros::Publisher pub_map;
    ros::Publisher pub_odom;
    ros::Publisher pub_poses;
    ros::Publisher pub_edge;
    ros::Publisher pub_surf;
    ros::Publisher pub_full;

    nav_msgs::Odometry odom_mapping;

    bool new_edge = false;
    bool new_surf = false;
    bool new_odom = false;
    bool new_each_odom = false;
    bool new_full_cloud = false;

    double time_new_edge;
    double time_new_surf;
    double time_new_odom;
    double time_new_each_odom = 0;

    pcl::PointCloud<PointType>::Ptr edge_last;
    pcl::PointCloud<PointType>::Ptr surf_last;
    pcl::PointCloud<PointType>::Ptr full_cloud;
    vector<pcl::PointCloud<PointType>::Ptr> full_clouds_ds;

    pcl::PointCloud<PointType>::Ptr edge_last_ds;
    pcl::PointCloud<PointType>::Ptr surf_last_ds;

    vector<pcl::PointCloud<PointType>::Ptr> edge_lasts_ds;
    vector<pcl::PointCloud<PointType>::Ptr> surf_lasts_ds;

    pcl::PointCloud<PointType>::Ptr edge_local_map;
    pcl::PointCloud<PointType>::Ptr surf_local_map;
    pcl::PointCloud<PointType>::Ptr edge_local_map_ds;
    pcl::PointCloud<PointType>::Ptr surf_local_map_ds;

    vector<pcl::PointCloud<PointType>::Ptr> vec_edge_cur_pts;
    vector<pcl::PointCloud<PointType>::Ptr> vec_edge_match_j;
    vector<pcl::PointCloud<PointType>::Ptr> vec_edge_match_l;

    vector<pcl::PointCloud<PointType>::Ptr> vec_surf_cur_pts;
    vector<pcl::PointCloud<PointType>::Ptr> vec_surf_normal;
    vector<vector<double>> vec_surf_scores;

    pcl::PointCloud<PointType>::Ptr latest_key_frames;
    pcl::PointCloud<PointType>::Ptr latest_key_frames_ds;
    pcl::PointCloud<PointType>::Ptr his_key_frames;
    pcl::PointCloud<PointType>::Ptr his_key_frames_ds;

    pcl::PointCloud<PointXYZI>::Ptr pose_cloud_frame; //position of keyframe 关键帧
    // Usage for PointPoseInfo
    // position: x, y, z
    // orientation: qw - w, qx - x, qy - y, qz - z
    pcl::PointCloud<PointPoseInfo>::Ptr pose_info_cloud_frame; //pose of keyframe 关键帧位姿

    // 储存每一个lidar帧，不仅仅是关键帧
    pcl::PointCloud<PointXYZI>::Ptr pose_each_frame; //position of each frame
    pcl::PointCloud<PointPoseInfo>::Ptr pose_info_each_frame; //pose of each frame

    // 用来作为目标的回环目标帧（即当前位姿）
    PointXYZI select_pose;
    PointType pt_in_local, pt_in_map;

    pcl::PointCloud<PointType>::Ptr global_map;
    pcl::PointCloud<PointType>::Ptr global_map_ds;

    vector<pcl::PointCloud<PointType>::Ptr> edge_frames;
    vector<pcl::PointCloud<PointType>::Ptr> surf_frames;

    deque<pcl::PointCloud<PointType>::Ptr> recent_edge_keyframes;
    deque<pcl::PointCloud<PointType>::Ptr> recent_surf_keyframes;
    int latest_frame_idx;

    pcl::KdTreeFLANN<PointType>::Ptr kd_tree_edge_local_map;
    pcl::KdTreeFLANN<PointType>::Ptr kd_tree_surf_local_map;
    pcl::KdTreeFLANN<PointXYZI>::Ptr kd_tree_his_key_poses;

    vector<int> pt_search_idx;
    vector<float> pt_search_sq_dists;

    pcl::VoxelGrid<PointType> ds_filter_edge;
    pcl::VoxelGrid<PointType> ds_filter_surf;
    pcl::VoxelGrid<PointType> ds_filter_edge_map;
    pcl::VoxelGrid<PointType> ds_filter_surf_map;
    pcl::VoxelGrid<PointType> ds_filter_his_frames;
    pcl::VoxelGrid<PointType> ds_filter_global_map;

    // 储存滑窗内 对应关键帧的 残差数量
    vector<int> vec_edge_res_cnt;
    vector<int> vec_surf_res_cnt;

    // Form of the transformation
    vector<double> abs_pose;
    vector<double> last_pose;

    mutex mutual_exclusion;

    int max_num_iter;

    // Boolean for functions
    bool loop_closure_on;

    gtsam::NonlinearFactorGraph glocal_pose_graph;
    gtsam::Values glocal_init_estimate;
    gtsam::ISAM2 *isam;
    gtsam::Values glocal_estimated;

    gtsam::noiseModel::Diagonal::shared_ptr prior_noise;
    gtsam::noiseModel::Diagonal::shared_ptr odom_noise;
    gtsam::noiseModel::Diagonal::shared_ptr constraint_noise;

    // Loop closure detection related
    bool loop_to_close;
    int closest_his_idx;
    int latest_frame_idx_loop;
    bool loop_closed;

    int local_map_width;

    double lc_search_radius;
    int lc_map_width;
    float lc_icp_thres;

    int slide_window_width;

    //index of keyframe
    vector<int> keyframe_idx;           // abs_pose变量的索引，即keyframe_idx[0] 表示第0个关键帧在abs_poses[]的索引， abs_poses.at(keyframe_idx[0])就是第0个关键帧的pose
    vector<int> keyframe_id_in_frame;   // 关键帧在所有lidar帧的idx

    vector<vector<double>> abs_poses;

    int num_kf_sliding; // 这个没用的变量

    vector<sensor_msgs::ImuConstPtr> imu_buf;
    nav_msgs::Odometry::ConstPtr odom_cur;
    vector<nav_msgs::Odometry::ConstPtr> each_odom_buf;     // 前端给出的当前帧到上一帧的相对位姿变换
    double time_last_imu;
    double cur_time_imu;    ///< 已经处理过的imu的时间戳
    bool first_imu;
    vector<Preintegration*> pre_integrations;
    Eigen::Vector3d acc_0, gyr_0, g, tmp_acc_0, tmp_gyr_0;

    Eigen::Vector3d tmp_P, tmp_V;
    Eigen::Quaterniond tmp_Q;

    vector<Eigen::Vector3d> Ps;
    vector<Eigen::Vector3d> Vs;
    vector<Eigen::Matrix3d> Rs;
    vector<Eigen::Vector3d> Bas;
    vector<Eigen::Vector3d> Bgs;
    vector<vector<double>> para_speed_bias;

    //extrinsic imu to lidar
    //imu -> lidar 的相对位姿变换
    Eigen::Quaterniond q_lb;
    Eigen::Vector3d t_lb;
    //lidar -> imu 的相对位姿变换
    Eigen::Quaterniond q_bl;
    Eigen::Vector3d t_bl;

    double ql2b_w, ql2b_x, ql2b_y, ql2b_z, tl2b_x, tl2b_y, tl2b_z;

    int idx_imu;    ///< 处理过的imu的索引

    //first sliding window optimazition
    bool first_opt;

    // for marginalization
    MarginalizationInfo *last_marginalization_info;
    vector<double *> last_marginalization_parameter_blocks;

    double **tmpQuat;
    double **tmpTrans;
    double **tmpSpeedBias;

    bool marg = true;

    vector<int> imu_idx_in_kf;  ///< 记录 时间戳 < 当前最新关键帧的IMU索引

    double time_last_loop = 0;

    bool quat_ini = false;

    string imu_topic;

    double surf_dist_thres;
    double kd_max_radius;
    bool save_pcd = false;


    double lidar_const = 0;
    double sum_thre = 0;
    int mapping_interval = 1;
    double lc_time_thres = 30.0;

    string frame_id = "lili_om_rot";
    string data_set;
    double runtime = 0;

public:
    BackendFusion(): nh("~") {
        initializeParameters();
        allocateMemory();

        sub_full_cloud = nh.subscribe<sensor_msgs::PointCloud2>("/full_point_cloud", 100, &BackendFusion::full_cloudHandler, this);
        sub_edge = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 100, &BackendFusion::edge_lastHandler, this);
        sub_surf = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 100, &BackendFusion::surfaceLastHandler, this);
        sub_odom = nh.subscribe<nav_msgs::Odometry>("/odom", 5, &BackendFusion::odomHandler, this);
        sub_each_odom = nh.subscribe<nav_msgs::Odometry>("/each_odom", 5, &BackendFusion::eachOdomHandler, this);

        sub_imu = nh.subscribe<sensor_msgs::Imu>(imu_topic, 200, &BackendFusion::imuHandler, this);

        pub_map = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_map", 2);
        pub_odom = nh.advertise<nav_msgs::Odometry>("/odom_mapped", 2);
        pub_poses = nh.advertise<sensor_msgs::PointCloud2>("/trajectory", 2);
        pub_edge = nh.advertise<sensor_msgs::PointCloud2>("/map_corner_less_sharp", 2);
        pub_surf = nh.advertise<sensor_msgs::PointCloud2>("/map_surf_less_flat", 2);
        pub_full = nh.advertise<sensor_msgs::PointCloud2>("/raw_scan", 2);
    }

    ~BackendFusion() {}

    void allocateMemory() {
        tmpQuat = new double *[slide_window_width];
        tmpTrans = new double *[slide_window_width];
        tmpSpeedBias = new double *[slide_window_width];
        for (int i = 0; i < slide_window_width; ++i) {
            tmpQuat[i] = new double[4];
            tmpTrans[i] = new double[3];
            tmpSpeedBias[i] = new double[9];
        }

        edge_last.reset(new pcl::PointCloud<PointType>());
        surf_last.reset(new pcl::PointCloud<PointType>());
        edge_local_map.reset(new pcl::PointCloud<PointType>());
        surf_local_map.reset(new pcl::PointCloud<PointType>());
        edge_last_ds.reset(new pcl::PointCloud<PointType>());
        surf_last_ds.reset(new pcl::PointCloud<PointType>());
        edge_local_map_ds.reset(new pcl::PointCloud<PointType>());
        surf_local_map_ds.reset(new pcl::PointCloud<PointType>());
        full_cloud.reset(new pcl::PointCloud<PointType>());

        for(int i = 0; i < slide_window_width; i++) {
            pcl::PointCloud<PointType>::Ptr tmpSurfCurrent;
            tmpSurfCurrent.reset(new pcl::PointCloud<PointType>());
            vec_surf_cur_pts.push_back(tmpSurfCurrent);

            vector<double> tmpD;
            vec_surf_scores.push_back(tmpD);

            pcl::PointCloud<PointType>::Ptr tmpSurfNorm;
            tmpSurfNorm.reset(new pcl::PointCloud<PointType>());
            vec_surf_normal.push_back(tmpSurfNorm);

            vec_surf_res_cnt.push_back(0);

            pcl::PointCloud<PointType>::Ptr tmpCornerCurrent;
            tmpCornerCurrent.reset(new pcl::PointCloud<PointType>());
            vec_edge_cur_pts.push_back(tmpCornerCurrent);

            pcl::PointCloud<PointType>::Ptr tmpCornerL;
            tmpCornerL.reset(new pcl::PointCloud<PointType>());
            vec_edge_match_l.push_back(tmpCornerL);

            pcl::PointCloud<PointType>::Ptr tmpCornerJ;
            tmpCornerJ.reset(new pcl::PointCloud<PointType>());
            vec_edge_match_j.push_back(tmpCornerJ);

            vec_edge_res_cnt.push_back(0);
        }

        pose_cloud_frame.reset(new pcl::PointCloud<PointXYZI>());
        pose_info_cloud_frame.reset(new pcl::PointCloud<PointPoseInfo>());

        pose_each_frame.reset(new pcl::PointCloud<PointXYZI>());
        pose_info_each_frame.reset(new pcl::PointCloud<PointPoseInfo>());

        global_map.reset(new pcl::PointCloud<PointType>());
        global_map_ds.reset(new pcl::PointCloud<PointType>());

        latest_key_frames.reset(new pcl::PointCloud<PointType>());
        latest_key_frames_ds.reset(new pcl::PointCloud<PointType>());
        his_key_frames.reset(new pcl::PointCloud<PointType>());
        his_key_frames_ds.reset(new pcl::PointCloud<PointType>());

        kd_tree_edge_local_map.reset(new pcl::KdTreeFLANN<PointType>());
        kd_tree_surf_local_map.reset(new pcl::KdTreeFLANN<PointType>());
        kd_tree_his_key_poses.reset(new pcl::KdTreeFLANN<PointXYZI>());
    }

    void initializeParameters() {
        gtsam::ISAM2Params isamPara;
        isamPara.relinearizeThreshold = 0.1;
        isamPara.relinearizeSkip = 1;
        isam = new gtsam::ISAM2(isamPara);

        // Load parameters from yaml
        if (!getParameter("/common/data_set", data_set)) {
            ROS_WARN("data_set not set, use default value: utbm");
            data_set = "utbm";
        }

        if (!getParameter("/backend_fusion/surf_dist_thres", surf_dist_thres)) {
            ROS_WARN("surf_dist_thres not set, use default value: 0.1");
            surf_dist_thres = 0.1;
        }

        if (!getParameter("/backend_fusion/kd_max_radius", kd_max_radius)) {
            ROS_WARN("kd_max_radius not set, use default value: 1.0");
            kd_max_radius = 1.0;
        }

        if (!getParameter("/backend_fusion/save_pcd", save_pcd)) {
            ROS_WARN("save_pcd not set, use default value: false");
            save_pcd = false;
        }

        if (!getParameter("/backend_fusion/mapping_interval", mapping_interval)) {
            ROS_WARN("mapping_interval not set, use default value: 1");
            mapping_interval = 1;
        }

        if (!getParameter("/backend_fusion/lc_time_thres", lc_time_thres)) {
            ROS_WARN("lc_time_thres not set, use default value: 30.0");
            lc_time_thres = 30.0;
        }
        // 激光约束权重
        if (!getParameter("/backend_fusion/lidar_const", lidar_const)) {
            ROS_WARN("lidar_const not set, use default value: 1.0");
            lidar_const = 1.0;
        }

        if (!getParameter("/backend_fusion/imu_topic", imu_topic)) {
            ROS_WARN("imu_topic not set, use default value: /imu/data");
            imu_topic = "/imu/data";
        }

        if (!getParameter("/backend_fusion/max_num_iter", max_num_iter)) {
            ROS_WARN("maximal iteration number of mapping optimization not set, use default value: 50");
            max_num_iter = 50;
        }

        if (!getParameter("/backend_fusion/loop_closure_on", loop_closure_on)) {
            ROS_WARN("loop closure detection set to false");
            loop_closure_on = false;
        }

        if (!getParameter("/backend_fusion/local_map_width", local_map_width)) {
            ROS_WARN("local_map_width not set, use default value: 5");
            local_map_width = 5;
        }

        if (!getParameter("/backend_fusion/lc_search_radius", lc_search_radius)) {
            ROS_WARN("lc_search_radius not set, use default value: 7.0");
            lc_search_radius = 7.0;
        }

        if (!getParameter("/backend_fusion/lc_map_width", lc_map_width)) {
            ROS_WARN("lc_map_width not set, use default value: 25");
            lc_map_width = 25;
        }

        if (!getParameter("/backend_fusion/lc_icp_thres", lc_icp_thres)) {
            ROS_WARN("lc_icp_thres not set, use default value: 0.3");
            lc_icp_thres = 0.3;
        }

        if (!getParameter("/backend_fusion/slide_window_width", slide_window_width)) {
            ROS_WARN("slide_window_width not set, use default value: 4");
            slide_window_width = 4;
        }

        //extrinsic parameters
        if (!getParameter("/backend_fusion/ql2b_w", ql2b_w))  {
            ROS_WARN("ql2b_w not set, use default value: 1");
            ql2b_w = 1;
        }

        if (!getParameter("/backend_fusion/ql2b_x", ql2b_x)) {
            ROS_WARN("ql2b_x not set, use default value: 0");
            ql2b_x = 0;
        }

        if (!getParameter("/backend_fusion/ql2b_y", ql2b_y)) {
            ROS_WARN("ql2b_y not set, use default value: 0");
            ql2b_y = 0;
        }

        if (!getParameter("/backend_fusion/ql2b_z", ql2b_z)) {
            ROS_WARN("ql2b_z not set, use default value: 0");
            ql2b_z = 0;
        }

        if (!getParameter("/backend_fusion/tl2b_x", tl2b_x)) {
            ROS_WARN("tl2b_x not set, use default value: 0");
            tl2b_x = 0;
        }

        if (!getParameter("/backend_fusion/tl2b_y", tl2b_y)) {
            ROS_WARN("tl2b_y not set, use default value: 0");
            tl2b_y = 0;
        }

        if (!getParameter("/backend_fusion/tl2b_z", tl2b_z)) {
            ROS_WARN("tl2b_z not set, use default value: 0");
            tl2b_z = 0;
        }

        last_marginalization_info = nullptr;
        tmp_P = tmp_V = Eigen::Vector3d(0, 0, 0);
        tmp_Q = Eigen::Quaterniond::Identity();
        idx_imu = 0;
        first_opt = false;
        cur_time_imu = -1;

        // 初始化时， Rs,Ps... 就push进了空元素
        Rs.push_back(Eigen::Matrix3d::Identity());
        Ps.push_back(Eigen::Vector3d::Zero());
        Vs.push_back(Eigen::Vector3d(0, 0, 0));

        Bas.push_back(Eigen::Vector3d::Zero());
        Bgs.push_back(Eigen::Vector3d(0, 0, 0));
        vector<double> tmpSpeedBias;
        tmpSpeedBias.push_back(0.0);
        tmpSpeedBias.push_back(0.0);
        tmpSpeedBias.push_back(0.0);
        tmpSpeedBias.push_back(0.0);
        tmpSpeedBias.push_back(0.0);
        tmpSpeedBias.push_back(0.0);
        tmpSpeedBias.push_back(0.0);
        tmpSpeedBias.push_back(0.0);
        tmpSpeedBias.push_back(0.0);

        para_speed_bias.push_back(tmpSpeedBias);

        num_kf_sliding = 0;
        time_last_imu = 0;
        first_imu = false;

        g = Eigen::Vector3d(0, 0, 9.805);

        time_new_edge = 0;
        time_new_surf = 0;
        time_new_odom = 0;

        abs_pose.push_back(1);
        last_pose.push_back(1);

        latest_frame_idx = 0;

        vector<double> tmpOdom;
        tmpOdom.push_back(1);

        for (int i = 1; i < 7; ++i) {
            abs_pose.push_back(0);
            last_pose.push_back(0);
            tmpOdom.push_back(0);
        }

        // abs_poses 也push进了空元素
        abs_poses.push_back(tmpOdom);

        abs_pose = tmpOdom;

        ds_filter_edge.setLeafSize(0.2, 0.2, 0.2);
        ds_filter_surf.setLeafSize(0.4, 0.4, 0.4);
        ds_filter_edge_map.setLeafSize(0.2, 0.2, 0.2);
        ds_filter_surf_map.setLeafSize(0.4, 0.4, 0.4);
        ds_filter_his_frames.setLeafSize(0.4, 0.4, 0.4);
        ds_filter_global_map.setLeafSize(0.2, 0.2, 0.2);

        odom_mapping.header.frame_id = frame_id;

        gtsam::Vector vector6p(6);
        gtsam::Vector vector6o(6);
        vector6p << 1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-8;
        vector6o << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4;
        prior_noise = gtsam::noiseModel::Diagonal::Variances(vector6p);
        odom_noise = gtsam::noiseModel::Diagonal::Variances(vector6o);

        loop_to_close = false;
        loop_closed = false;

        q_lb = Eigen::Quaterniond(ql2b_w, ql2b_x, ql2b_y, ql2b_z);
        t_lb = Eigen::Vector3d(tl2b_x, tl2b_y, tl2b_z);

        q_bl = q_lb.inverse();
        t_bl = - (q_bl * t_lb);
    }

    void full_cloudHandler(const sensor_msgs::PointCloud2ConstPtr& pointCloudIn) {
        full_cloud->clear();
        pcl::fromROSMsg(*pointCloudIn, *full_cloud);
        pcl::PointCloud<PointType>::Ptr full(new pcl::PointCloud<PointType>());
        pcl::copyPointCloud(*full_cloud, *full);
        new_full_cloud = true;
    }

    void edge_lastHandler(const sensor_msgs::PointCloud2ConstPtr& pointCloudIn) {
        edge_last->clear();
        time_new_edge = pointCloudIn->header.stamp.toSec();
        pcl::fromROSMsg(*pointCloudIn, *edge_last);

        new_edge = true;
    }

    void surfaceLastHandler(const sensor_msgs::PointCloud2ConstPtr& pointCloudIn) {
        surf_last->clear();
        time_new_surf = pointCloudIn->header.stamp.toSec();
        pcl::fromROSMsg(*pointCloudIn, *surf_last);
        new_surf = true;
    }

    void odomHandler(const nav_msgs::Odometry::ConstPtr& odomIn) {
        //cout<<"odom_sub_cnt: "<<++odom_sub_cnt<<endl;

        time_new_odom = odomIn->header.stamp.toSec();
        odom_cur = odomIn;

        new_odom = true;
    }

    void eachOdomHandler(const nav_msgs::Odometry::ConstPtr& odomIn) {
        time_new_each_odom = odomIn->header.stamp.toSec();
        each_odom_buf.push_back(odomIn);

        if(each_odom_buf.size() > 50)
            each_odom_buf[each_odom_buf.size() - 51] = nullptr;

        new_each_odom = true;
    }

    void imuHandler(const sensor_msgs::ImuConstPtr& ImuIn) {
        time_last_imu = ImuIn->header.stamp.toSec();

        imu_buf.push_back(ImuIn);

        if(imu_buf.size() > 600)
            imu_buf[imu_buf.size() - 601] = nullptr;

        // 第一帧
        if (cur_time_imu < 0)
            cur_time_imu = time_last_imu;

        // 不是第一帧
        if (!first_imu) {
            if(data_set == "utbm")
                g = Eigen::Vector3d(ImuIn->linear_acceleration.x, ImuIn->linear_acceleration.y, ImuIn->linear_acceleration.z);
            else {
                // 取9轴姿态
                Eigen::Quaterniond quat(ImuIn->orientation.w,
                                        ImuIn->orientation.x,
                                        ImuIn->orientation.y,
                                        ImuIn->orientation.z);

                Rs[0] = quat.toRotationMatrix();
                abs_poses[0][0] = ImuIn->orientation.w;
                abs_poses[0][1] = ImuIn->orientation.x;
                abs_poses[0][2] = ImuIn->orientation.y;
                abs_poses[0][3] = ImuIn->orientation.z;
            }

            first_imu = true;
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            dx = ImuIn->linear_acceleration.x;
            dy = ImuIn->linear_acceleration.y;
            dz = ImuIn->linear_acceleration.z;
            rx = ImuIn->angular_velocity.x;
            ry = ImuIn->angular_velocity.y;
            rz = ImuIn->angular_velocity.z;

            Eigen::Vector3d linear_acceleration(dx, dy, dz);
            Eigen::Vector3d angular_velocity(rx, ry, rz);
            acc_0 = linear_acceleration;
            gyr_0 = angular_velocity;
            pre_integrations.push_back(new Preintegration(acc_0, gyr_0, Bas[0], Bgs[0]));
            /// 这个重力向量?
            pre_integrations.back()->g_vec_ = -g;
        }
    }

    // 将点变换到世界坐标系
    void transformPoint(PointType const *const pi, PointType *const po) {
        Eigen::Quaterniond quaternion(abs_pose[0],
                abs_pose[1],
                abs_pose[2],
                abs_pose[3]);
        Eigen::Vector3d transition(abs_pose[4],
                abs_pose[5],
                abs_pose[6]);

        Eigen::Vector3d ptIn(pi->x, pi->y, pi->z);
        Eigen::Vector3d ptOut = quaternion * ptIn + transition;


        po->x = ptOut.x();
        po->y = ptOut.y();
        po->z = ptOut.z();
        po->intensity = pi->intensity;
    }

    void transformPoint(PointType const *const pi, PointType *const po, Eigen::Quaterniond quaternion, Eigen::Vector3d transition) {
        Eigen::Vector3d ptIn(pi->x, pi->y, pi->z);
        Eigen::Vector3d ptOut = quaternion * ptIn + transition;

        po->x = ptOut.x();
        po->y = ptOut.y();
        po->z = ptOut.z();
        po->intensity = pi->intensity;
    }

    pcl::PointCloud<PointType>::Ptr transformCloud(const pcl::PointCloud<PointType>::Ptr &cloudIn) {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        int numPts = cloudIn->points.size();
        cloudOut->resize(numPts);

        for (int i = 0; i < numPts; ++i) {
            PointType ptIn = cloudIn->points[i];
            PointType ptOut;
            transformPoint(&ptIn, &ptOut);
            cloudOut->points[i] = ptOut;
        }
        return cloudOut;
    }

    pcl::PointCloud<PointType>::Ptr transformCloud(const pcl::PointCloud<PointType>::Ptr &cloudIn, PointPoseInfo * PointInfoIn) {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        Eigen::Quaterniond quaternion(PointInfoIn->qw,
                                      PointInfoIn->qx,
                                      PointInfoIn->qy,
                                      PointInfoIn->qz);
        Eigen::Vector3d transition(PointInfoIn->x,
                                   PointInfoIn->y,
                                   PointInfoIn->z);

        int numPts = cloudIn->points.size();
        cloudOut->resize(numPts);

        for (int i = 0; i < numPts; ++i) {
            Eigen::Vector3d ptIn(cloudIn->points[i].x, cloudIn->points[i].y, cloudIn->points[i].z);
            Eigen::Vector3d ptOut = quaternion * ptIn + transition;

            PointType pt;
            pt.x = ptOut.x();
            pt.y = ptOut.y();
            pt.z = ptOut.z();
            pt.intensity = cloudIn->points[i].intensity;

            cloudOut->points[i] = pt;
        }

        return cloudOut;
    }


    pcl::PointCloud<PointType>::Ptr transformCloud(const pcl::PointCloud<PointType>::Ptr &cloudIn, Eigen::Quaterniond quaternion, Eigen::Vector3d transition) {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        int numPts = cloudIn->points.size();
        cloudOut->resize(numPts);

        for (int i = 0; i < numPts; ++i) {
            Eigen::Vector3d ptIn(cloudIn->points[i].x, cloudIn->points[i].y, cloudIn->points[i].z);
            Eigen::Vector3d ptOut = quaternion * ptIn + transition;

            PointType pt;
            pt.x = ptOut.x();
            pt.y = ptOut.y();
            pt.z = ptOut.z();
            pt.intensity = cloudIn->points[i].intensity;

            cloudOut->points[i] = pt;
        }

        return cloudOut;
    }

    // 输入一帧imu数据以及delta t
    // 更新 Rs.back() Ps.back() Vs.back() acc_0 ,gyro_0
    void processIMU(double dt, const Eigen::Vector3d &linear_acceleration, const Eigen::Vector3d &angular_velocity) {
        /// 插入了新的帧，需要push_back新的数据
        if(pre_integrations.size() < abs_poses.size()) {
            pre_integrations.push_back(new Preintegration(acc_0, gyr_0, Bas.back(), Bgs.back()));
            pre_integrations.back()->g_vec_ = -g;
            Bas.push_back(Bas.back());
            Bgs.push_back(Bgs.back());
            Rs.push_back(Rs.back());
            Ps.push_back(Ps.back());
            Vs.push_back(Vs.back());
        }
        /// 如果没有插入新的帧，则只更新最后的数据
        // 姿态更新前的加速度 （去除重力）
        Eigen::Vector3d un_acc_0 = Rs.back() * (acc_0 - Bas.back()) - g;
        // 姿态更新
        Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs.back();
        Rs.back() *= deltaQ(un_gyr * dt).toRotationMatrix();
        // 姿态更新后的加速度 （去除重力）
        Eigen::Vector3d un_acc_1 = Rs.back() * (linear_acceleration - Bas.back()) - g;
        Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        // 位置、速度更新
        Ps.back() += dt * Vs.back() + 0.5 * dt * dt * un_acc;
        Vs.back() += dt * un_acc;

        // 调用Preintegration::push_back()函数，而非队列的push_back
        // 储存dt，acc，gyro数据
        // 然后传播
        pre_integrations.back()->push_back(dt, linear_acceleration, angular_velocity);

        // 加速度更新
        acc_0 = linear_acceleration;
        // 角速度更新
        gyr_0 = angular_velocity;
    }

    void optimizeSlidingWindowWithLandMark() {
        if(slide_window_width < 1) return;
        // 如果关键帧数量比滑窗宽度小，不优化
        if(keyframe_idx.size() < slide_window_width) return;

        first_opt = true;

        // keyframe_idx[keyframe_idx.size()-1] : 最新关键帧对应的 abs_pose 的索引
        // keyframe_idx[keyframe_idx.size()-slide_window_width] : 滑窗最旧关键帧对应的 abs_pose 的索引
        int windowSize = keyframe_idx[keyframe_idx.size()-1] - keyframe_idx[keyframe_idx.size()-slide_window_width] + 1;
        //std::cout<<(windowSize == slide_window_width)<<std::endl;

        // 构造局部地图 kd-tree
        kd_tree_surf_local_map->setInputCloud(surf_local_map_ds);
        kd_tree_edge_local_map->setInputCloud(edge_local_map_ds);


        for (int iterCount = 0; iterCount < 1; ++iterCount) {
            ceres::LossFunction *lossFunction = new ceres::CauchyLoss(1.0);
            ceres::LocalParameterization *quatParameterization = new ceres::QuaternionParameterization();
            ceres::Problem problem;

            //eigen to double
            // keyframe_idx[keyframe_idx.size()-slide_window_width] : 滑窗最旧关键帧对应的 abs_pose 的索引
            // keyframe_idx[keyframe_idx.size()-slide_window_width] : i的起始
            for (int i = keyframe_idx[keyframe_idx.size()-slide_window_width]; i <= keyframe_idx.back(); i++){
                // tmpQuat: 滑窗大小的二维数组，储存滑窗中关键帧的四元数姿态
                Eigen::Quaterniond tmpQ(Rs[i]);
                tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][0] = tmpQ.w();
                tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][1] = tmpQ.x();
                tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][2] = tmpQ.y();
                tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][3] = tmpQ.z();
                tmpTrans[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][0] = Ps[i][0];
                tmpTrans[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][1] = Ps[i][1];
                tmpTrans[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][2] = Ps[i][2];

                // i = keyframe_idx[..]
                // keyframe_idx[] 储存 关键帧 对应的 abs_pose 在 abs_poses队列中的索引
                abs_poses[i][0] = tmpQ.w();
                abs_poses[i][1] = tmpQ.x();
                abs_poses[i][2] = tmpQ.y();
                abs_poses[i][3] = tmpQ.z();
                abs_poses[i][4] = Ps[i][0];
                abs_poses[i][5] = Ps[i][1];
                abs_poses[i][6] = Ps[i][2];

                for(int j = 0; j < 9; j++) {
                    tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][j] = para_speed_bias[i][j];
                }

                // 添加参数块，滑窗内的关键帧位姿、imu bias

                //add lidar parameters
                problem.AddParameterBlock(tmpTrans[i-keyframe_idx[keyframe_idx.size()-slide_window_width]], 3);
                problem.AddParameterBlock(tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]], 4, quatParameterization);

                //add IMU parameters
                //imu bias
                problem.AddParameterBlock(tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]], 9);
            }

            abs_pose = abs_poses.back();

            if(true) {
                // 如果有上一次marg的数据，则添加到残差块
                if (last_marginalization_info) {
                    // construct new marginlization_factor
                    MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                    problem.AddResidualBlock(marginalization_factor, NULL,
                                             last_marginalization_parameter_blocks);
                }
            }

            if(!marg) {
                // 如果发生回环,marg被设置为false
                // 则添加 imu 的 bias先验
                //add prior factor
                for(int i = 0; i < slide_window_width - 1; i++) {

                    vector<double> tmps;
                    for(int j = 0; j < 9; j++) {
                        tmps.push_back(tmpSpeedBias[i][j]);
                    }
                    ceres::CostFunction *speedBiasPriorFactor = SpeedBiasPriorFactorAutoDiff::Create(tmps);
                    problem.AddResidualBlock(speedBiasPriorFactor, NULL, tmpSpeedBias[i]);
                }

            }

            // 添加imu 预积分因子
            // keyframe_idx[keyframe_idx.size()-slide_window_width] : 滑窗最前面那一帧对应的 abs_pose 在 abs_poses中的索引
            for (int idx = keyframe_idx[keyframe_idx.size()-slide_window_width]; idx < keyframe_idx.back(); ++idx) {
                //add imu factor
                ImuFactor *imuFactor = new ImuFactor(pre_integrations[idx+1]);
                // 添加预积分因子， 约束的参数有： 滑窗相邻两帧的位姿、speedbias
                problem.AddResidualBlock(imuFactor, NULL,
                                         tmpTrans[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]],
                        tmpQuat[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]],
                        tmpSpeedBias[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]],
                        tmpTrans[idx+1-keyframe_idx[keyframe_idx.size()-slide_window_width]],
                        tmpQuat[idx+1-keyframe_idx[keyframe_idx.size()-slide_window_width]],
                        tmpSpeedBias[idx+1-keyframe_idx[keyframe_idx.size()-slide_window_width]]);
            }

            // 添加激光雷达特征关联残差
            for (int idx = keyframe_idx[keyframe_idx.size()-slide_window_width]; idx <= keyframe_idx.back(); idx++) {
                // 取滑窗中的关键帧位姿
                Eigen::Quaterniond Q2 = Eigen::Quaterniond(tmpQuat[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]][0],
                        tmpQuat[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]][1],
                        tmpQuat[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]][2],
                        tmpQuat[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]][3]);
                Eigen::Vector3d T2 = Eigen::Vector3d(tmpTrans[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]][0],
                        tmpTrans[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]][1],
                        tmpTrans[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]][2]);

                Q2 = Q2 * q_lb.inverse();
                T2 = T2 - Q2 * t_lb;

                // idVec: 0,1,2,3 ....
                int idVec = idx - keyframe_idx[keyframe_idx.size()-slide_window_width];
                // 如果局部地图点数足够
                if (surf_local_map_ds->points.size() > 50 && edge_local_map_ds->points.size() > 0) {
                    // 特征关联
                    findCorrespondingSurfFeatures(idx-1, Q2, T2);
                    findCorrespondingCornerFeatures(idx-1, Q2, T2);

                    for (int i = 0; i < vec_edge_res_cnt[idVec]; ++i) {
                        Eigen::Vector3d currentPt(vec_edge_cur_pts[idVec]->points[i].x,
                                                  vec_edge_cur_pts[idVec]->points[i].y,
                                                  vec_edge_cur_pts[idVec]->points[i].z);
                        Eigen::Vector3d lastPtJ(vec_edge_match_j[idVec]->points[i].x,
                                                vec_edge_match_j[idVec]->points[i].y,
                                                vec_edge_match_j[idVec]->points[i].z);
                        Eigen::Vector3d lastPtL(vec_edge_match_l[idVec]->points[i].x,
                                                vec_edge_match_l[idVec]->points[i].y,
                                                vec_edge_match_l[idVec]->points[i].z);

                        // pt_in_local.intensity = lidar_const;
                        ceres::CostFunction *costFunction = LidarEdgeFactor::Create(currentPt, lastPtJ, lastPtL, q_lb, t_lb, vec_edge_cur_pts[idVec]->points[i].intensity * 200 / vec_edge_res_cnt[idVec]);


                        //new ceres::AutoDiffCostFunction<LidarEdgeFactor, 残差维度, 参数块1维度, 参数块2维度>
                        problem.AddResidualBlock(costFunction, lossFunction, tmpTrans[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]],
                                tmpQuat[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]]);
                    }

                    for (int i = 0; i < vec_surf_res_cnt[idVec]; ++i) {
                        Eigen::Vector3d currentPt(vec_surf_cur_pts[idVec]->points[i].x,
                                                  vec_surf_cur_pts[idVec]->points[i].y,
                                                  vec_surf_cur_pts[idVec]->points[i].z);
                        Eigen::Vector3d norm(vec_surf_normal[idVec]->points[i].x,
                                             vec_surf_normal[idVec]->points[i].y,
                                             vec_surf_normal[idVec]->points[i].z);
                        double normInverse = vec_surf_normal[idVec]->points[i].intensity;

                        //LidarPlaneNormAnalyticFactor *costFunction = new LidarPlaneNormAnalyticFactor(currentPt, norm, normInverse);
                        ceres::CostFunction *costFunction = LidarPlaneNormFactor::Create(currentPt, norm, q_lb, t_lb, normInverse, vec_surf_scores[idVec][i] * 1000 / vec_surf_res_cnt[idVec]);


                        problem.AddResidualBlock(costFunction, lossFunction, tmpTrans[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]],
                                tmpQuat[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]]);

                    }
                }
                else
                    ROS_WARN("Not enough feature points from the map");

            }


            // 求解
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_QR;
            options.max_num_iterations = max_num_iter;
            options.minimizer_progress_to_stdout = false;
            options.check_gradients = false;
            options.gradient_check_relative_precision = 0.5;

            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);

            // 对求解的姿态进行归一化
            for(int i = 0; i < windowSize; i++) {
                if(tmpQuat[i][0] < 0) {
                    Eigen::Quaterniond tmp(tmpQuat[i][0],
                            tmpQuat[i][1],
                            tmpQuat[i][2],
                            tmpQuat[i][3]);
                    tmp = unifyQuaternion(tmp);
                    tmpQuat[i][0] = tmp.w();
                    tmpQuat[i][1] = tmp.x();
                    tmpQuat[i][2] = tmp.y();
                    tmpQuat[i][3] = tmp.z();
                }
            }
        }

        MarginalizationInfo *marginalization_info = new MarginalizationInfo();

        // 如果之前有marg，则向新的marg信息中添加之前marg的信息
        if (last_marginalization_info) {
            vector<int> drop_set;
            // 如果之前marg信息中保留的部分 与 准备要marg的滑窗最老帧有关联，那么之前的marg信息中保留的部分需要被marg
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++) {
                if (last_marginalization_parameter_blocks[i] == tmpTrans[0] ||
                        last_marginalization_parameter_blocks[i] == tmpQuat[0] ||
                        last_marginalization_parameter_blocks[i] == tmpSpeedBias[0])
                    drop_set.push_back(i);
            }
            // construct new marginlization_factor
            // 构造新的marg因子
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);

            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set);

            marginalization_info->AddResidualBlockInfo(residual_block_info);
        }

        // 如果发生回环,marg被设置为false
        if(!marg) {
            //add prior factor
            // 遍历滑窗
            for(int i = 0; i < slide_window_width - 1; i++) {

                vector<double*> tmp;
                tmp.push_back(tmpTrans[i]);
                tmp.push_back(tmpQuat[i]);

                //drop_set 表示要marg哪些变量
                vector<int> drop_set;
                if(i == 0) {
                    drop_set.push_back(0);
                    drop_set.push_back(1);
                }

                vector<double> tmps;
                for(int j = 0; j < 9; j++) {
                    tmps.push_back(tmpSpeedBias[i][j]);
                }

                vector<double*> tmp1;
                tmp1.push_back(tmpSpeedBias[i]);

                vector<int> drop_set1;
                if(i == 0) {
                    drop_set1.push_back(0);
                }
                // 添加
                ceres::CostFunction *speedBiasPriorFactor = SpeedBiasPriorFactorAutoDiff::Create(tmps);
                ResidualBlockInfo *residual_block_info1 = new ResidualBlockInfo(speedBiasPriorFactor, NULL,
                                                                                tmp1,
                                                                                drop_set1);

                marginalization_info->AddResidualBlockInfo(residual_block_info1);
            }

            marg = true;
        }


        //imu
        ImuFactor *imuFactor = new ImuFactor(pre_integrations[keyframe_idx[keyframe_idx.size()-slide_window_width]+1]);

        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imuFactor, NULL,
                                                                       vector<double *>{        //imu因子
                                                                                                tmpTrans[0],
                                                                                                tmpQuat[0],
                                                                                                tmpSpeedBias[0],
                                                                                                tmpTrans[1],
                                                                                                tmpQuat[1],
                                                                                                tmpSpeedBias[1]
                                                                       },
                                                                       vector<int>{0, 1, 2});   //要marg的变量

        marginalization_info->AddResidualBlockInfo(residual_block_info);


        //lidar
        for (int idx = keyframe_idx[keyframe_idx.size()-slide_window_width]; idx <= keyframe_idx.back(); idx++) {
            ceres::LossFunction *lossFunction = new ceres::CauchyLoss(1.0);
            int idVec = idx - keyframe_idx[keyframe_idx.size()-slide_window_width];
            if (surf_local_map_ds->points.size() > 50 && edge_local_map_ds->points.size() > 0) {
                vector<double*> tmp;
                tmp.push_back(tmpTrans[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]]);
                tmp.push_back(tmpQuat[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]]);

                for (int i = 0; i < vec_surf_res_cnt[idVec]; ++i) {
                    Eigen::Vector3d currentPt(vec_surf_cur_pts[idVec]->points[i].x,
                                              vec_surf_cur_pts[idVec]->points[i].y,
                                              vec_surf_cur_pts[idVec]->points[i].z);
                    Eigen::Vector3d norm(vec_surf_normal[idVec]->points[i].x,
                                         vec_surf_normal[idVec]->points[i].y,
                                         vec_surf_normal[idVec]->points[i].z);
                    double normInverse = vec_surf_normal[idVec]->points[i].intensity;

                    //LidarPlaneNormAnalyticFactor *costFunction = new LidarPlaneNormAnalyticFactor(currentPt, norm, normInverse);
                    ceres::CostFunction *costFunction = LidarPlaneNormFactor::Create(currentPt, norm, q_lb, t_lb, normInverse, vec_surf_scores[idVec][i] * 1000 / vec_surf_res_cnt[idVec]);

                    vector<int> drop_set;
                    if(idx == keyframe_idx[keyframe_idx.size()-slide_window_width]) {
                        drop_set.push_back(0);
                        drop_set.push_back(1);
                    }
                    ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(costFunction, lossFunction,
                                                                                   tmp,
                                                                                   drop_set);
                    marginalization_info->AddResidualBlockInfo(residual_block_info);
                }


                for (int i = 0; i < vec_edge_res_cnt[idVec]; ++i) {
                    Eigen::Vector3d currentPt(vec_edge_cur_pts[idVec]->points[i].x,
                                              vec_edge_cur_pts[idVec]->points[i].y,
                                              vec_edge_cur_pts[idVec]->points[i].z);
                    Eigen::Vector3d lastPtJ(vec_edge_match_j[idVec]->points[i].x,
                                            vec_edge_match_j[idVec]->points[i].y,
                                            vec_edge_match_j[idVec]->points[i].z);
                    Eigen::Vector3d lastPtL(vec_edge_match_l[idVec]->points[i].x,
                                            vec_edge_match_l[idVec]->points[i].y,
                                            vec_edge_match_l[idVec]->points[i].z);


                    ceres::CostFunction *costFunction = LidarEdgeFactor::Create(currentPt, lastPtJ, lastPtL, q_lb, t_lb, vec_edge_cur_pts[idVec]->points[i].intensity * 200 / vec_edge_res_cnt[idVec]);



                    vector<int> drop_set;
                    if(idx == keyframe_idx[keyframe_idx.size()-slide_window_width]) {
                        drop_set.push_back(0);
                        drop_set.push_back(1);
                    }
                    ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(costFunction, lossFunction,
                                                                                   tmp,
                                                                                   drop_set);
                    marginalization_info->AddResidualBlockInfo(residual_block_info);
                }

            }
            else {
                ROS_WARN("Not enough feature points from the map");
            }


            vec_edge_cur_pts[idVec]->clear();
            vec_edge_match_j[idVec]->clear();
            vec_edge_match_l[idVec]->clear();
            vec_edge_res_cnt[idVec] = 0;

            vec_surf_cur_pts[idVec]->clear();
            vec_surf_normal[idVec]->clear();
            vec_surf_res_cnt[idVec] = 0;
            vec_surf_scores[idVec].clear();
        }

        marginalization_info->PreMarginalize();
        marginalization_info->Marginalize();

        // marg完成

        // 地址映射地址
        // addr_shift [ 滑窗第i帧地址 ] =  滑窗第i-1帧地址
        // 相当于滑窗内数据向前移
        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i < windowSize; ++i) {
            addr_shift[reinterpret_cast<long>(tmpTrans[i])] = tmpTrans[i-1];
            addr_shift[reinterpret_cast<long>(tmpQuat[i])] = tmpQuat[i-1];
            addr_shift[reinterpret_cast<long>(tmpSpeedBias[i])] = tmpSpeedBias[i-1];
        }

        vector<double *> parameter_blocks = marginalization_info->GetParameterBlocks(addr_shift);


        if (last_marginalization_info) {
            delete last_marginalization_info;
        }
        // marg后的信息
        last_marginalization_info = marginalization_info;
        // 要保留的变量的在下一个H矩阵构造时的地址
        // (相当于做了一步，数据向前移)
        last_marginalization_parameter_blocks = parameter_blocks;


        //double to eigen
        //遍历滑动窗口
        for (int i = keyframe_idx[keyframe_idx.size()-slide_window_width]; i <= keyframe_idx.back(); ++i){
            // 优化前后做差
            double dp0 = Ps[i][0] - tmpTrans[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][0];
            double dp1 = Ps[i][1] - tmpTrans[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][1];
            double dp2 = Ps[i][2] - tmpTrans[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][2];
            double pnorm = sqrt(dp0*dp0+dp1*dp1+dp2*dp2);

            double dv0 = Vs[i][0] - tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][0];
            double dv1 = Vs[i][1] - tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][1];
            double dv2 = Vs[i][2] - tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][2];
            double vnorm = sqrt(dv0*dv0+dv1*dv1+dv2*dv2);

            double dba1 = para_speed_bias[i][3] - tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][3];
            double dba2 = para_speed_bias[i][4] - tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][4];
            double dba3 = para_speed_bias[i][5] - tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][5];
            double dbg1 = para_speed_bias[i][6] - tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][6];
            double dbg2 = para_speed_bias[i][7] - tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][7];
            double dbg3 = para_speed_bias[i][8] - tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][8];

            Eigen::Quaterniond dq = Eigen::Quaterniond (tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][0],
                    tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][1],
                    tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][2],
                    tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][3]).normalized().inverse() *
                    Eigen::Quaterniond(Rs[i]);
            double qnorm = dq.vec().norm();

            // 检查是否超过范围
            if(pnorm < 10) {
                // 取优化后的结果进行更新
                abs_poses[i][4] = tmpTrans[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][0];
                abs_poses[i][5] = tmpTrans[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][1];
                abs_poses[i][6] = tmpTrans[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][2];

                Ps[i][0] = tmpTrans[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][0];
                Ps[i][1] = tmpTrans[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][1];
                Ps[i][2] = tmpTrans[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][2];
            } else
                ROS_WARN("bad optimization result of p!!!!!!!!!!!!!");

            if(qnorm < 10) {
                abs_poses[i][0] = tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][0];
                abs_poses[i][1] = tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][1];
                abs_poses[i][2] = tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][2];
                abs_poses[i][3] = tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][3];

                Rs[i] = Eigen::Quaterniond (tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][0],
                        tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][1],
                        tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][2],
                        tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][3]).normalized().toRotationMatrix();
            } else
                ROS_WARN("bad optimization result of q!!!!!!!!!!!!!");

            // 速度更新
            if(vnorm < 10) {
                for(int j = 0; j < 3; j++) {
                    para_speed_bias[i][j] = tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][j];
                }
                Vs[i][0] = tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][0];
                Vs[i][1] = tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][1];
                Vs[i][2] = tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][2];
            } else
                ROS_WARN("bad optimization result of v!!!!!!!!!!!!!");

            // bias更新
            if(abs(dba1) < 22) {
                para_speed_bias[i][3] = tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][3];
                Bas[i][0] = para_speed_bias[i][3];
            } else
                ROS_WARN("bad ba1!!!!!!!!!!");

            if(abs(dba2) < 22) {
                para_speed_bias[i][4] = tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][4];
                Bas[i][1] = para_speed_bias[i][4];
            } else
                ROS_WARN("bad ba2!!!!!!!!!!");

            if(abs(dba3) < 22) {
                para_speed_bias[i][5] = tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][5];
                Bas[i][2] = para_speed_bias[i][5];
            } else
                ROS_WARN("bad ba3!!!!!!!!!!");

            if(abs(dbg1) < 22) {
                para_speed_bias[i][6] = tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][6];
                Bgs[i][0] = para_speed_bias[i][6];
            } else
                ROS_WARN("bad bg1!!!!!!!!!!");

            if(abs(dbg2) < 22) {
                para_speed_bias[i][7] = tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][7];
                Bgs[i][1] = para_speed_bias[i][7];
            } else
                ROS_WARN("bad bg2!!!!!!!!!!");

            if(abs(dbg3) < 22) {
                para_speed_bias[i][8] = tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][8];
                Bgs[i][2] = para_speed_bias[i][8];
            } else
                ROS_WARN("bad bg3!!!!!!!!!!");
        }

        updatePose();
    }

    // 取优化的结果来更新abs_pose 和 pose_cloud_frame , pose_info_cloud_frame
    void updatePose() {
        abs_pose = abs_poses.back();
        for (int i = keyframe_idx[keyframe_idx.size()-slide_window_width]; i <= keyframe_idx[keyframe_idx.size()-1]; ++i){
            pose_cloud_frame->points[i-1].x = abs_poses[i][4];
            pose_cloud_frame->points[i-1].y = abs_poses[i][5];
            pose_cloud_frame->points[i-1].z = abs_poses[i][6];

            pose_info_cloud_frame->points[i-1].x = abs_poses[i][4];
            pose_info_cloud_frame->points[i-1].y = abs_poses[i][5];
            pose_info_cloud_frame->points[i-1].z = abs_poses[i][6];
            pose_info_cloud_frame->points[i-1].qw = abs_poses[i][0];
            pose_info_cloud_frame->points[i-1].qx = abs_poses[i][1];
            pose_info_cloud_frame->points[i-1].qy = abs_poses[i][2];
            pose_info_cloud_frame->points[i-1].qz = abs_poses[i][3];
        }
    }


    void optimizeLocalGraph(vector<double*> paraEach) {
        ceres::LocalParameterization *quatParameterization = new ceres::QuaternionParameterization();
        ceres::Problem problem;

        // ||: 关键帧
        //  |: 普通帧
        ///      || - | - | - | - ||
        /// id    2   3   4   5   6
        /// numPara = 6-2-1 = 3
        /// 表示一共有3帧的位姿需要优化,即中间的3帧

        // keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]: 滑窗最旧关键帧的前一关键帧对应id
        // keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width] : 滑窗最旧关键帧对应id
        int numPara = keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width] - keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1;
        double dQuat[numPara][4];
        double dTrans[numPara][3];

        for(int i = keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1] + 1;
            i < keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width]; i++) {
            dTrans[i-keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1][0] = pose_each_frame->points[i].x;
            dTrans[i-keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1][1] = pose_each_frame->points[i].y;
            dTrans[i-keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1][2] = pose_each_frame->points[i].z;

            dQuat[i-keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1][0] = pose_info_each_frame->points[i].qw;
            dQuat[i-keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1][1] = pose_info_each_frame->points[i].qx;
            dQuat[i-keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1][2] = pose_info_each_frame->points[i].qy;
            dQuat[i-keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1][3] = pose_info_each_frame->points[i].qz;

            problem.AddParameterBlock(dTrans[i-keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1], 3);
            problem.AddParameterBlock(dQuat[i-keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1], 4, quatParameterization);
        }


        /// paraEach[]: 后一帧到前一帧的变换
        /// paraEach[]: dP, dq, dP, dq ...

        // 滑窗最旧关键帧的前一关键帧的后一普通帧 到 滑窗最旧关键帧的前一关键帧的约束关系
        // 待求解变量是： 滑窗最旧关键帧的前一关键帧的后一普通帧
        ceres::CostFunction *LeftFactor = LidarPoseLeftFactorAutoDiff::Create(
                Eigen::Quaterniond(paraEach[1][0], paraEach[1][1], paraEach[1][2], paraEach[1][3]),     // delta_q
                Eigen::Vector3d(paraEach[0][0], paraEach[0][1], paraEach[0][2]),                        // delta_p
                Eigen::Quaterniond(pose_info_cloud_frame->points[pose_cloud_frame->points.size() - slide_window_width - 1].qw,  //滑窗最旧关键帧的前一关键帧的位姿（滑窗优化得到的）
                                   pose_info_cloud_frame->points[pose_cloud_frame->points.size() - slide_window_width - 1].qx,
                                   pose_info_cloud_frame->points[pose_cloud_frame->points.size() - slide_window_width - 1].qy,
                                   pose_info_cloud_frame->points[pose_cloud_frame->points.size() - slide_window_width - 1].qz),
                Eigen::Vector3d(pose_info_cloud_frame->points[pose_cloud_frame->points.size() - slide_window_width - 1].x,
                                pose_info_cloud_frame->points[pose_cloud_frame->points.size() - slide_window_width - 1].y,
                                pose_info_cloud_frame->points[pose_cloud_frame->points.size() - slide_window_width - 1].z));
        problem.AddResidualBlock(LeftFactor, NULL, dTrans[0], dQuat[0]);

        // 滑窗最旧关键帧的前一关键帧 到 滑窗最旧关键帧 之间的普通帧的约束关系
        for(int i = 0; i < numPara - 1; i++) {
            ceres::CostFunction *Factor = LidarPoseFactorAutoDiff::Create(Eigen::Quaterniond(paraEach[2*i+1][0], paraEach[2*i+1][1], paraEach[2*i+1][2], paraEach[2*i+1][3]),
                    Eigen::Vector3d(paraEach[2*i][0], paraEach[2*i][1], paraEach[2*i][2]));
            problem.AddResidualBlock(Factor, NULL, dTrans[i], dQuat[i], dTrans[i+1], dQuat[i+1]);

        }

        // 滑窗最旧关键帧 到 滑窗最旧关键帧的前一普通帧 的约束关系
        ceres::CostFunction *RightFactor = LidarPoseRightFactorAutoDiff::Create(
                Eigen::Quaterniond(paraEach.back()[0], paraEach.back()[1], paraEach.back()[2], paraEach.back()[3]),                 //delta P
                Eigen::Vector3d(paraEach[paraEach.size()-2][0], paraEach[paraEach.size()-2][1], paraEach[paraEach.size()-2][2]),    //delta q
                Eigen::Quaterniond(pose_info_cloud_frame->points[pose_cloud_frame->points.size() - slide_window_width].qw,          //滑窗最旧关键帧位姿（滑窗优化得到）
                                   pose_info_cloud_frame->points[pose_cloud_frame->points.size() - slide_window_width].qx,
                                   pose_info_cloud_frame->points[pose_cloud_frame->points.size() - slide_window_width].qy,
                                   pose_info_cloud_frame->points[pose_cloud_frame->points.size() - slide_window_width].qz),
                Eigen::Vector3d(pose_info_cloud_frame->points[pose_cloud_frame->points.size() - slide_window_width].x,
                                pose_info_cloud_frame->points[pose_cloud_frame->points.size() - slide_window_width].y,
                                pose_info_cloud_frame->points[pose_cloud_frame->points.size() - slide_window_width].z));
        problem.AddResidualBlock(RightFactor, NULL, dTrans[numPara-1], dQuat[numPara-1]);


        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.trust_region_strategy_type = ceres::DOGLEG;
        options.max_num_iterations = 15;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        // 更新参数
        for(int i = keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1] + 1;
            i < keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width]; i++) {
            pose_each_frame->points[i].x = dTrans[i-keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1][0];
            pose_each_frame->points[i].y = dTrans[i-keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1][1];
            pose_each_frame->points[i].z = dTrans[i-keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1][2];

            pose_info_each_frame->points[i].x = dTrans[i-keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1][0];
            pose_info_each_frame->points[i].y = dTrans[i-keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1][1];
            pose_info_each_frame->points[i].z = dTrans[i-keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1][2];
            pose_info_each_frame->points[i].qw = dQuat[i-keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1][0];
            pose_info_each_frame->points[i].qx = dQuat[i-keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1][1];
            pose_info_each_frame->points[i].qy = dQuat[i-keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1][2];
            pose_info_each_frame->points[i].qz = dQuat[i-keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1][3];
        }
    }


    void buildLocalMapWithLandMark() {
        // Initialization
        // 初始化, pose_cloud_frame 为空
        // 将点变换到base_link坐标系，以第一帧base_link坐标系作为世界(局部?)坐标系?
        if (pose_cloud_frame->points.size() < 1) {
            PointPoseInfo Tbl;
            Tbl.qw = q_bl.w();
            Tbl.qx = q_bl.x();
            Tbl.qy = q_bl.y();
            Tbl.qz = q_bl.z();
            Tbl.x = t_bl.x();
            Tbl.y = t_bl.y();
            Tbl.z = t_bl.z();
            //ROS_INFO("Initialization for local map building");
            *edge_local_map += *transformCloud(edge_last, &Tbl);
            *surf_local_map += *transformCloud(surf_last, &Tbl);
            return;
        }

        // 如果recent局部地图的size还不够，则从后向前遍历关键帧构建局部地图
        if (recent_surf_keyframes.size() < local_map_width) {
            /// 为啥要清空
            recent_edge_keyframes.clear();
            recent_surf_keyframes.clear();

            // 从后向前遍历关键帧
            for (int i = pose_cloud_frame->points.size() - 1; i >= 0; --i) {
                //.intensity = pose_cloud_frame->points.size();
                int idx = (int)pose_cloud_frame->points[i].intensity;

                Eigen::Quaterniond q_po(pose_info_cloud_frame->points[idx].qw,
                                        pose_info_cloud_frame->points[idx].qx,
                                        pose_info_cloud_frame->points[idx].qy,
                                        pose_info_cloud_frame->points[idx].qz);

                Eigen::Vector3d t_po(pose_info_cloud_frame->points[idx].x,
                                     pose_info_cloud_frame->points[idx].y,
                                     pose_info_cloud_frame->points[idx].z);

                Eigen::Quaterniond q_tmp = q_po * q_bl;
                Eigen::Vector3d t_tmp = q_po * t_bl + t_po;

                PointPoseInfo Ttmp;
                Ttmp.qw = q_tmp.w();
                Ttmp.qx = q_tmp.x();
                Ttmp.qy = q_tmp.y();
                Ttmp.qz = q_tmp.z();
                Ttmp.x = t_tmp.x();
                Ttmp.y = t_tmp.y();
                Ttmp.z = t_tmp.z();

                // 将特征点变换到全局坐标系上，然后保存
                recent_edge_keyframes.push_front(transformCloud(edge_frames[idx], &Ttmp));
                recent_surf_keyframes.push_front(transformCloud(surf_frames[idx], &Ttmp));

                if (recent_surf_keyframes.size() >= local_map_width)
                    break;
            }
        }
        // If already more then 50 frames, pop the frames at the beginning
        // 如果recent局部地图size>目标，则弹出，然后推入最新
        else {
            if (latest_frame_idx != pose_cloud_frame->points.size() - 1) {
                recent_edge_keyframes.pop_front();
                recent_surf_keyframes.pop_front();
                latest_frame_idx = pose_cloud_frame->points.size() - 1;

                Eigen::Quaterniond q_po(pose_info_cloud_frame->points[latest_frame_idx].qw,
                                        pose_info_cloud_frame->points[latest_frame_idx].qx,
                                        pose_info_cloud_frame->points[latest_frame_idx].qy,
                                        pose_info_cloud_frame->points[latest_frame_idx].qz);

                Eigen::Vector3d t_po(pose_info_cloud_frame->points[latest_frame_idx].x,
                                     pose_info_cloud_frame->points[latest_frame_idx].y,
                                     pose_info_cloud_frame->points[latest_frame_idx].z);

                Eigen::Quaterniond q_tmp = q_po * q_bl;
                Eigen::Vector3d t_tmp = q_po * t_bl + t_po;

                PointPoseInfo Ttmp;
                Ttmp.qw = q_tmp.w();
                Ttmp.qx = q_tmp.x();
                Ttmp.qy = q_tmp.y();
                Ttmp.qz = q_tmp.z();
                Ttmp.x = t_tmp.x();
                Ttmp.y = t_tmp.y();
                Ttmp.z = t_tmp.z();

                recent_edge_keyframes.push_back(transformCloud(edge_frames[latest_frame_idx], &Ttmp));
                recent_surf_keyframes.push_back(transformCloud(surf_frames[latest_frame_idx], &Ttmp));
            }
        }

        // 构建局部地图
        for (int i = 0; i < recent_surf_keyframes.size(); ++i) {
            *edge_local_map += *recent_edge_keyframes[i];
            *surf_local_map += *recent_surf_keyframes[i];
        }
    }

    void downSampleCloud() {
        // 局部特征地图降采样
        ds_filter_surf_map.setInputCloud(surf_local_map);
        ds_filter_surf_map.filter(*surf_local_map_ds);

        ds_filter_edge_map.setInputCloud(edge_local_map);
        ds_filter_edge_map.filter(*edge_local_map_ds);

        // 全点云地图降采样
        pcl::PointCloud<PointType>::Ptr fullDS(new pcl::PointCloud<PointType>());
        ds_filter_surf_map.setInputCloud(full_cloud);
        ds_filter_surf_map.filter(*fullDS);
        full_clouds_ds.push_back(fullDS);

        // 对当前帧扫描的特征降采样
        surf_last_ds->clear();
        ds_filter_surf.setInputCloud(surf_last);
        ds_filter_surf.filter(*surf_last_ds);
        pcl::PointCloud<PointType>::Ptr surf(new pcl::PointCloud<PointType>());
        pcl::copyPointCloud(*surf_last_ds, *surf);
        surf_lasts_ds.push_back(surf);

        edge_last_ds->clear();
        ds_filter_edge.setInputCloud(edge_last);
        ds_filter_edge.filter(*edge_last_ds);
        pcl::PointCloud<PointType>::Ptr corner(new pcl::PointCloud<PointType>());
        pcl::copyPointCloud(*edge_last_ds, *corner);
        edge_lasts_ds.push_back(corner);
    }


    void findCorrespondingCornerFeatures(int idx, Eigen::Quaterniond q, Eigen::Vector3d t) {
        int idVec = idx - keyframe_idx[keyframe_idx.size()-slide_window_width] + 1;
        vec_edge_res_cnt[idVec] = 0;
        // 遍历关键帧边缘点
        for (int i = 0; i < edge_lasts_ds[idx]->points.size(); ++i) {
            pt_in_local = edge_lasts_ds[idx]->points[i];

            transformPoint(&pt_in_local, &pt_in_map, q, t);
            kd_tree_edge_local_map->nearestKSearch(pt_in_map, 5, pt_search_idx, pt_search_sq_dists);

            if (pt_search_sq_dists[4] < 1.0) {
                std::vector<Eigen::Vector3d> nearCorners;
                Eigen::Vector3d center(0, 0, 0);
                for (int j = 0; j < 5; ++j) {
                    Eigen::Vector3d pt(edge_local_map_ds->points[pt_search_idx[j]].x,
                            edge_local_map_ds->points[pt_search_idx[j]].y,
                            edge_local_map_ds->points[pt_search_idx[j]].z);
                    center = center + pt;
                    nearCorners.push_back(pt);
                }
                center /= 5.0;

                // 协方差
                // Covariance matrix of distance error
                Eigen::Matrix3d matA1 = Eigen::Matrix3d::Zero();

                for (int j = 0; j < 5; ++j) {
                    Eigen::Vector3d zeroMean = nearCorners[j] - center;
                    matA1 = matA1 + zeroMean * zeroMean.transpose();
                }

                // Computes eigenvalues and eigenvectors of selfadjoint matrices
                // The eigenvalues re sorted in increasing order
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigenSolver(matA1);

                Eigen::Vector3d unitDirection = eigenSolver.eigenvectors().col(2);

                // if one eigenvalue is significantly larger than the other two
                if (eigenSolver.eigenvalues()[2] > 3 * eigenSolver.eigenvalues()[1]) {
                    Eigen::Vector3d ptOnLine = center;
                    Eigen::Vector3d ptA, ptB;
                    ptA = ptOnLine + 0.1 * unitDirection;
                    ptB = ptOnLine - 0.1 * unitDirection;
                    pt_in_local.intensity = lidar_const;
                    Eigen::Vector3d lp(pt_in_map.x, pt_in_map.y, pt_in_map.z);

                    Eigen::Vector3d nu = (lp - ptA).cross(lp - ptB);
                    Eigen::Vector3d de = ptA - ptB;

                    double dist = nu.norm() / de.norm();
                    if(dist < 0.1) {
                        PointType pointA, pointB;
                        pointA.x = ptA.x();
                        pointA.y = ptA.y();
                        pointA.z = ptA.z();
                        pointB.x = ptB.x();
                        pointB.y = ptB.y();
                        pointB.z = ptB.z();
                        // 储存边缘线上的两个点
                        vec_edge_cur_pts[idVec]->push_back(pt_in_local);
                        vec_edge_match_j[idVec]->push_back(pointA);
                        vec_edge_match_l[idVec]->push_back(pointB);

                        ++vec_edge_res_cnt[idVec];
                    }

                }
            }

        }
    }
    // 根据输入的预估位姿，查找平面特征关联
    // 输入的 idx 是
    void findCorrespondingSurfFeatures(int idx, Eigen::Quaterniond q, Eigen::Vector3d t) {
        int idVec = idx - keyframe_idx[keyframe_idx.size()-slide_window_width] + 1;
        vec_surf_res_cnt[idVec] = 0;
        // 遍历关键帧的平面点
        for (int i = 0; i < surf_lasts_ds[idx]->points.size(); ++i) {
            pt_in_local = surf_lasts_ds[idx]->points[i];
            // 将变换到世界坐标系
            transformPoint(&pt_in_local, &pt_in_map, q, t);
            // 在局部地图找最近5个点
            kd_tree_surf_local_map->nearestKSearch(pt_in_map, 5, pt_search_idx, pt_search_sq_dists);

            Eigen::Matrix<double, 5, 3> matA0;
            Eigen::Matrix<double, 5, 1> matB0 = - Eigen::Matrix<double, 5, 1>::Ones();
            if (pt_search_sq_dists[4] < kd_max_radius) {
                for (int j = 0; j < 5; ++j) {
                    matA0(j, 0) = surf_local_map_ds->points[pt_search_idx[j]].x;
                    matA0(j, 1) = surf_local_map_ds->points[pt_search_idx[j]].y;
                    matA0(j, 2) = surf_local_map_ds->points[pt_search_idx[j]].z;
                }
                // 求法向量
                // Get the norm of the plane using linear solver based on QR composition
                Eigen::Vector3d norm = matA0.colPivHouseholderQr().solve(matB0);
                double normInverse = 1 / norm.norm();
                norm.normalize(); // get the unit norm

                // Make sure that the plan is fit
                // 检查平面是否符合
                bool planeValid = true;
                for (int j = 0; j < 5; ++j) {
                    if (fabs(norm.x() * surf_local_map_ds->points[pt_search_idx[j]].x +
                             norm.y() * surf_local_map_ds->points[pt_search_idx[j]].y +
                             norm.z() * surf_local_map_ds->points[pt_search_idx[j]].z + normInverse) > surf_dist_thres) {
                        planeValid = false;
                        break;
                    }
                }

                // if one eigenvalue is significantly larger than the other two
                // 计算权重
                if (planeValid) {
                    float pd = norm.x() * pt_in_map.x + norm.y() * pt_in_map.y + norm.z() *pt_in_map.z + normInverse;
                    float weight = 1 - 0.9 * fabs(pd) / sqrt(sqrt(pt_in_map.x * pt_in_map.x + pt_in_map.y * pt_in_map.y + pt_in_map.z * pt_in_map.z));

                    if(weight > 0.3) {
                        PointType normal;
                        normal.x = weight * norm.x();
                        normal.y = weight * norm.y();
                        normal.z = weight * norm.z();
                        normal.intensity = weight * normInverse;
                        // 保存特征关联信息
                        vec_surf_cur_pts[idVec]->push_back(pt_in_local);
                        vec_surf_normal[idVec]->push_back(normal);

                        ++vec_surf_res_cnt[idVec];
                        // 权重？
                        vec_surf_scores[idVec].push_back(lidar_const*weight);
                    }
                }
            }
        }
    }

    void saveKeyFramesAndFactors() {
        // 保存当前位姿(这里主要是先占一个坑，后面再更新值)
        abs_poses.push_back(abs_pose);
        // 最新关键帧在所有lidar帧的idx
        keyframe_id_in_frame.push_back(each_odom_buf.size()-1);

        pcl::PointCloud<PointType>::Ptr cornerEachFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surfEachFrame(new pcl::PointCloud<PointType>());

        pcl::copyPointCloud(*edge_last_ds, *cornerEachFrame);
        pcl::copyPointCloud(*surf_last_ds, *surfEachFrame);

        // 保存当前帧降采样后的特征点
        edge_frames.push_back(cornerEachFrame);
        surf_frames.push_back(surfEachFrame);

        // record index of kayframe on imu preintegration poses
        // 记录当前最新关键帧索引 （与imu 预积分位姿对应的）
        // keyframe_idx[0] 表示第0个关键帧在abs_poses[]的索引， abs_poses.at(keyframe_idx[0])就是第0个关键帧的pose
        keyframe_idx.push_back(abs_poses.size()-1);
        /***************************************
        ** qpc Debug:
        ****************************************/
        //std::cout<<abs_poses.size()-1<<std::endl;
        /***************************************
        ** Debug Finish!
        ** 输出：
        ** 1
        ** 2
        ** ...
        ** 结论： keyframe_idx[]第0个元素对应
        **       abs_poses[]第1个元素，相当于滞后1个
        ** 因为在初始化时， abs_poses就填了一个空的坑位，
        ** 第一次进入到这个函数，运行到这里的时候， abs_poses.size() == 2
        ****************************************/

        double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;

        int i = idx_imu;
        Eigen::Quaterniond tmpOrient;
        // 取当前关键帧时间戳
        double timeodom_cur = odom_cur->header.stamp.toSec();
        // 如果imu队列中已经处理过的imu数据时间戳 > 当前关键帧, 不合理
        if(imu_buf[i]->header.stamp.toSec() > timeodom_cur)
            ROS_WARN("Timestamp not synchronized, please check your hardware!");

        // 开始遍历imu队列，对当前关键帧之前的imu数据进行处理
        while(imu_buf[i]->header.stamp.toSec() < timeodom_cur) {
            // 取时间戳
            double t = imu_buf[i]->header.stamp.toSec();
            // 第一帧
            if (cur_time_imu < 0)
                cur_time_imu = t;
            // 计算与上一帧imu的时间差
            double dt = t - cur_time_imu;
            cur_time_imu = imu_buf[i]->header.stamp.toSec();
            dx = imu_buf[i]->linear_acceleration.x;
            dy = imu_buf[i]->linear_acceleration.y;
            dz = imu_buf[i]->linear_acceleration.z;
            // 限幅
            if(dx > 15.0) dx = 15.0;
            if(dy > 15.0) dy = 15.0;
            if(dz > 18.0) dz = 18.0;

            if(dx < -15.0) dx = -15.0;
            if(dy < -15.0) dy = -15.0;
            if(dz < -18.0) dz = -18.0;

            rx = imu_buf[i]->angular_velocity.x;
            ry = imu_buf[i]->angular_velocity.y;
            rz = imu_buf[i]->angular_velocity.z;

            tmpOrient = Eigen::Quaterniond(imu_buf[i]->orientation.w,
                                           imu_buf[i]->orientation.x,
                                           imu_buf[i]->orientation.y,
                                           imu_buf[i]->orientation.z);
            // 输入一帧imu数据以及delta t
            // 更新 Rs.back() Ps.back() Vs.back() acc_0 ,gyro_0
            processIMU(dt, Eigen::Vector3d(dx, dy, dz), Eigen::Vector3d(rx, ry, rz));
            i++;
            if(i >= imu_buf.size())
                break;
        }

        // 到最新关键帧的imu idx
        imu_idx_in_kf.push_back(i - 1);

        // 后面还有imu数据，进行一次插值，得到刚好在 当前关键帧的数据
        if(i < imu_buf.size()) {
            double dt1 = timeodom_cur - cur_time_imu;
            double dt2 = imu_buf[i]->header.stamp.toSec() - timeodom_cur;

            double w1 = dt2 / (dt1 + dt2);
            double w2 = dt1 / (dt1 + dt2);

            Eigen::Quaterniond orient1 = Eigen::Quaterniond(imu_buf[i]->orientation.w,
                                                            imu_buf[i]->orientation.x,
                                                            imu_buf[i]->orientation.y,
                                                            imu_buf[i]->orientation.z);
            tmpOrient = tmpOrient.slerp(w2, orient1);

            dx = w1 * dx + w2 * imu_buf[i]->linear_acceleration.x;
            dy = w1 * dy + w2 * imu_buf[i]->linear_acceleration.y;
            dz = w1 * dz + w2 * imu_buf[i]->linear_acceleration.z;

            if(dx > 15.0) dx = 15.0;
            if(dy > 15.0) dy = 15.0;
            if(dz > 18.0) dz = 18.0;

            if(dx < -15.0) dx = -15.0;
            if(dy < -15.0) dy = -15.0;
            if(dz < -18.0) dz = -18.0;

            rx = w1 * rx + w2 * imu_buf[i]->angular_velocity.x;
            ry = w1 * ry + w2 * imu_buf[i]->angular_velocity.y;
            rz = w1 * rz + w2 * imu_buf[i]->angular_velocity.z;
            processIMU(dt1, Eigen::Vector3d(dx, dy, dz), Eigen::Vector3d(rx, ry, rz));
        }
        cur_time_imu = timeodom_cur;

        // ???
        vector<double> tmpSpeedBias;
        tmpSpeedBias.push_back(Vs.back().x());
        tmpSpeedBias.push_back(Vs.back().y());
        tmpSpeedBias.push_back(Vs.back().z());
        tmpSpeedBias.push_back(Bas.back().x());
        tmpSpeedBias.push_back(Bas.back().y());
        tmpSpeedBias.push_back(Bas.back().z());
        tmpSpeedBias.push_back(Bgs.back().x());
        tmpSpeedBias.push_back(Bgs.back().y());
        tmpSpeedBias.push_back(Bgs.back().z());
        para_speed_bias.push_back(tmpSpeedBias);
        idx_imu = i;

        // 保存关键帧位置 到 pose_cloud_frame ， 可是这还没优化求解呢。。。 （后面滑窗优化的时候会更新这个值）
        PointXYZI latestPose;
        PointPoseInfo latestPoseInfo;
        latestPose.x = Ps.back().x();
        latestPose.y = Ps.back().y();
        latestPose.z = Ps.back().z();
        latestPose.intensity = pose_cloud_frame->points.size();
        pose_cloud_frame->push_back(latestPose);

        // 保存关键帧位姿 到 pose_info_cloud_frame
        latestPoseInfo.x = Ps.back().x();
        latestPoseInfo.y = Ps.back().y();
        latestPoseInfo.z = Ps.back().z();
        Eigen::Quaterniond qs_last(Rs.back());
        latestPoseInfo.qw = qs_last.w();
        latestPoseInfo.qx = qs_last.x();
        latestPoseInfo.qy = qs_last.y();
        latestPoseInfo.qz = qs_last.z();
        latestPoseInfo.idx = pose_cloud_frame->points.size();
        latestPoseInfo.time = time_new_odom;

        pose_info_cloud_frame->push_back(latestPoseInfo);
        /***************************************
        ** 结论： pose_cloud_frame 、pose_info_cloud_frame
        **       与keyframe_idx[]元素顺序对应
        ****************************************/

        //optimize sliding window
        num_kf_sliding++;
        // 滑窗关键帧数量>1 或者 还没进行第一次 滑窗优化
        // 一定会进去的
        if(num_kf_sliding >= 1 || !first_opt) {
            optimizeSlidingWindowWithLandMark();
            num_kf_sliding = 0;
        }

        // 关键帧数量 == 滑窗width , 只会遇到一次
        if (pose_cloud_frame->points.size() == slide_window_width) {
            // 直接保存第0个关键帧就好了
            // 取最旧的关键帧，保存到pose_each_frame,pose_info_each_frame
            pose_each_frame->push_back(pose_cloud_frame->points[0]);
            pose_info_each_frame->push_back(pose_info_cloud_frame->points[0]);
        } else if(pose_cloud_frame->points.size() > slide_window_width) {
            // 如果关键帧数量 > 滑窗width
            // 否则，需要优化滑窗最旧帧的前一关键帧到滑窗最旧帧之间的普通帧位姿

            // ii: 到滑窗最旧帧的前一关键帧的imu索引
            int ii = imu_idx_in_kf[imu_idx_in_kf.size() - slide_window_width - 1];
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;

            /***************************************
            ** qpc Debug:
            ****************************************/
            //std::cout<<Ps.size()<<" "<<abs_poses.size()<<" "<<pose_cloud_frame->points.size()<<std::endl;
            /***************************************
            ** Debug Finish!
            ** 输出：
            ** 5 5 4
            ** 6 6 5
            ** 7 7 6
            ** 8 8 7
            ** 9 9 8
            ** 10 10 9
            ** abs_poses[1] 与 Ps[1] ...    <===> 对应于 pose_cloud_frame[0]
            ****************************************/

            //Ps.size() - slide_window_width : 滑窗最旧关键帧的前一关键帧
            Eigen::Vector3d Ptmp = Ps[Ps.size() - slide_window_width];
            Eigen::Vector3d Vtmp = Vs[Ps.size() - slide_window_width];
            Eigen::Matrix3d Rtmp = Rs[Ps.size() - slide_window_width];
            Eigen::Vector3d Batmp = Eigen::Vector3d::Zero();
            Eigen::Vector3d Bgtmp = Eigen::Vector3d::Zero();
            /***************************************
            ** qpc Debug:
            ****************************************/
            //std::cout<<Ps.size() - slide_window_width<<" "<<pose_cloud_frame->points.size() - slide_window_width - 1<<std::endl;
            /***************************************
            ** Debug Finish!
            ** 输出：
            ** 2 0
            ** 3 1
            ** 4 2
            ** 5 3 ...
            ** Ps[]的第2个元素
            ** 因为，slide_window_width==3，所以第3次才进到这里
            ****************************************/

            // i表示普通激光帧的序号
            // 从滑窗最旧帧的前一关键帧的后一个普通激光帧开始
            // 直到滑窗最旧关键帧
            for(int i = keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1] + 1;
                i < keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width]; i++) {

                // each_odom_buf[i-1]->header.stamp.toSec() ： 普通激光帧时间戳
                double dt1 = each_odom_buf[i-1]->header.stamp.toSec() - imu_buf[ii]->header.stamp.toSec();
                double dt2 = imu_buf[ii+1]->header.stamp.toSec() - each_odom_buf[i-1]->header.stamp.toSec();
                double w1 = dt2 / (dt1 + dt2);
                double w2 = dt1 / (dt1 + dt2);
                dx = w1 * imu_buf[ii]->linear_acceleration.x + w2 * imu_buf[ii+1]->linear_acceleration.x;
                dy = w1 * imu_buf[ii]->linear_acceleration.y + w2 * imu_buf[ii+1]->linear_acceleration.y;
                dz = w1 * imu_buf[ii]->linear_acceleration.z + w2 * imu_buf[ii+1]->linear_acceleration.z;

                rx = w1 * imu_buf[ii]->angular_velocity.x + w2 * imu_buf[ii+1]->angular_velocity.x;
                ry = w1 * imu_buf[ii]->angular_velocity.y + w2 * imu_buf[ii+1]->angular_velocity.y;
                rz = w1 * imu_buf[ii]->angular_velocity.z + w2 * imu_buf[ii+1]->angular_velocity.z;
                Eigen::Vector3d a0(dx, dy, dz);
                Eigen::Vector3d gy0(rx, ry, rz);
                ii++;
                // 积分开始时间
                double integStartTime = each_odom_buf[i-1]->header.stamp.toSec();

                // 从普通激光帧开始积分，直到下一个普通激光帧
                while(imu_buf[ii]->header.stamp.toSec() < each_odom_buf[i]->header.stamp.toSec()) {
                    double t = imu_buf[ii]->header.stamp.toSec();
                    double dt = t - integStartTime;
                    integStartTime = imu_buf[ii]->header.stamp.toSec();
                    dx = imu_buf[ii]->linear_acceleration.x;
                    dy = imu_buf[ii]->linear_acceleration.y;
                    dz = imu_buf[ii]->linear_acceleration.z;

                    rx = imu_buf[ii]->angular_velocity.x;
                    ry = imu_buf[ii]->angular_velocity.y;
                    rz = imu_buf[ii]->angular_velocity.z;

                    if(dx > 15.0) dx = 15.0;
                    if(dy > 15.0) dy = 15.0;
                    if(dz > 18.0) dz = 18.0;

                    if(dx < -15.0) dx = -15.0;
                    if(dy < -15.0) dy = -15.0;
                    if(dz < -18.0) dz = -18.0;

                    Eigen::Vector3d a1(dx, dy, dz);
                    Eigen::Vector3d gy1(rx, ry, rz);

                    Eigen::Vector3d un_acc_0 = Rtmp * (a0 - Batmp) - g;
                    Eigen::Vector3d un_gyr = 0.5 * (gy0 + gy1) - Bgtmp;
                    Rtmp *= deltaQ(un_gyr * dt).toRotationMatrix();
                    Eigen::Vector3d un_acc_1 = Rtmp * (a1 - Batmp) - g;
                    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
                    Ptmp += dt * Vtmp + 0.5 * dt * dt * un_acc;
                    Vtmp += dt * un_acc;

                    a0 = a1;
                    gy0 = gy1;

                    ii++;
                }

                dt1 = each_odom_buf[i]->header.stamp.toSec() - imu_buf[ii-1]->header.stamp.toSec();
                dt2 = imu_buf[ii]->header.stamp.toSec() - each_odom_buf[i]->header.stamp.toSec();
                w1 = dt2 / (dt1 + dt2);
                w2 = dt1 / (dt1 + dt2);
                dx = w1 * dx + w2 * imu_buf[ii]->linear_acceleration.x;
                dy = w1 * dy + w2 * imu_buf[ii]->linear_acceleration.y;
                dz = w1 * dz + w2 * imu_buf[ii]->linear_acceleration.z;

                rx = w1 * rx + w2 * imu_buf[ii]->angular_velocity.x;
                ry = w1 * ry + w2 * imu_buf[ii]->angular_velocity.y;
                rz = w1 * rz + w2 * imu_buf[ii]->angular_velocity.z;

                if(dx > 15.0) dx = 15.0;
                if(dy > 15.0) dy = 15.0;
                if(dz > 18.0) dz = 18.0;

                if(dx < -15.0) dx = -15.0;
                if(dy < -15.0) dy = -15.0;
                if(dz < -18.0) dz = -18.0;

                Eigen::Vector3d a1(dx, dy, dz);
                Eigen::Vector3d gy1(rx, ry, rz);

                Eigen::Vector3d un_acc_0 = Rtmp * (a0 - Batmp) - g;
                Eigen::Vector3d un_gyr = 0.5 * (gy0 + gy1) - Bgtmp;
                Rtmp *= deltaQ(un_gyr * dt1).toRotationMatrix();
                Eigen::Vector3d un_acc_1 = Rtmp * (a1 - Batmp) - g;
                Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
                Ptmp += dt1 * Vtmp + 0.5 * dt1 * dt1 * un_acc;
                Vtmp += dt1 * un_acc;

                ii--;

                /// 到这里，上面通过imu解算得到 下一个普通激光帧的位姿

                // 保存期间的每一个激光帧(普通帧)位姿
                Eigen::Quaterniond qqq(Rtmp);

                PointXYZI latestPose;
                PointPoseInfo latestPoseInfo;
                latestPose.x = Ptmp.x();
                latestPose.y = Ptmp.y();
                latestPose.z = Ptmp.z();
                pose_each_frame->push_back(latestPose);

                latestPoseInfo.x = Ptmp.x();
                latestPoseInfo.y = Ptmp.y();
                latestPoseInfo.z = Ptmp.z();
                latestPoseInfo.qw = qqq.w();
                latestPoseInfo.qx = qqq.x();
                latestPoseInfo.qy = qqq.y();
                latestPoseInfo.qz = qqq.z();
                latestPoseInfo.time = each_odom_buf[i]->header.stamp.toSec();
                pose_info_each_frame->push_back(latestPoseInfo);
            }
            /// 到这里， imu解算得到滑窗最旧关键帧之前的普通帧的位姿

            // pose_cloud_frame->points[pose_cloud_frame->points.size() - slide_window_width]: 最旧关键帧的位姿(滑窗优化得到的)
            pose_each_frame->push_back(pose_cloud_frame->points[pose_cloud_frame->points.size() - slide_window_width]);
            pose_info_each_frame->push_back(pose_info_cloud_frame->points[pose_cloud_frame->points.size() - slide_window_width]);

            // 取滑窗最旧关键帧的普通帧id
            int j = keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width];
            // 继续积分
            double dt1 = each_odom_buf[j-1]->header.stamp.toSec() - imu_buf[ii]->header.stamp.toSec();
            double dt2 = imu_buf[ii+1]->header.stamp.toSec() - each_odom_buf[j-1]->header.stamp.toSec();
            double w1 = dt2 / (dt1 + dt2);
            double w2 = dt1 / (dt1 + dt2);
            dx = w1 * imu_buf[ii]->linear_acceleration.x + w2 * imu_buf[ii+1]->linear_acceleration.x;
            dy = w1 * imu_buf[ii]->linear_acceleration.y + w2 * imu_buf[ii+1]->linear_acceleration.y;
            dz = w1 * imu_buf[ii]->linear_acceleration.z + w2 * imu_buf[ii+1]->linear_acceleration.z;

            rx = w1 * imu_buf[ii]->angular_velocity.x + w2 * imu_buf[ii+1]->angular_velocity.x;
            ry = w1 * imu_buf[ii]->angular_velocity.y + w2 * imu_buf[ii+1]->angular_velocity.y;
            rz = w1 * imu_buf[ii]->angular_velocity.z + w2 * imu_buf[ii+1]->angular_velocity.z;

            if(dx > 15.0) dx = 15.0;
            if(dy > 15.0) dy = 15.0;
            if(dz > 18.0) dz = 18.0;

            if(dx < -15.0) dx = -15.0;
            if(dy < -15.0) dy = -15.0;
            if(dz < -18.0) dz = -18.0;

            Eigen::Vector3d a0(dx, dy, dz);
            Eigen::Vector3d gy0(rx, ry, rz);
            ii++;
            double integStartTime = each_odom_buf[j-1]->header.stamp.toSec();

            while(imu_buf[ii]->header.stamp.toSec() < each_odom_buf[j]->header.stamp.toSec()) {
                double t = imu_buf[ii]->header.stamp.toSec();
                double dt = t - integStartTime;
                integStartTime = imu_buf[ii]->header.stamp.toSec();
                dx = imu_buf[ii]->linear_acceleration.x;
                dy = imu_buf[ii]->linear_acceleration.y;
                dz = imu_buf[ii]->linear_acceleration.z;

                rx = imu_buf[ii]->angular_velocity.x;
                ry = imu_buf[ii]->angular_velocity.y;
                rz = imu_buf[ii]->angular_velocity.z;

                if(dx > 15.0) dx = 15.0;
                if(dy > 15.0) dy = 15.0;
                if(dz > 18.0) dz = 18.0;

                if(dx < -15.0) dx = -15.0;
                if(dy < -15.0) dy = -15.0;
                if(dz < -18.0) dz = -18.0;

                Eigen::Vector3d a1(dx, dy, dz);
                Eigen::Vector3d gy1(rx, ry, rz);

                Eigen::Vector3d un_acc_0 = Rtmp * (a0 - Batmp) - g;
                Eigen::Vector3d un_gyr = 0.5 * (gy0 + gy1) - Bgtmp;
                Rtmp *= deltaQ(un_gyr * dt).toRotationMatrix();
                Eigen::Vector3d un_acc_1 = Rtmp * (a1 - Batmp) - g;
                Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
                Ptmp += dt * Vtmp + 0.5 * dt * dt * un_acc;
                Vtmp += dt * un_acc;

                a0 = a1;
                gy0 = gy1;

                ii++;
            }

            dt1 = each_odom_buf[j]->header.stamp.toSec() - imu_buf[ii-1]->header.stamp.toSec();
            dt2 = imu_buf[ii]->header.stamp.toSec() - each_odom_buf[j]->header.stamp.toSec();
            w1 = dt2 / (dt1 + dt2);
            w2 = dt1 / (dt1 + dt2);
            dx = w1 * dx + w2 * imu_buf[ii]->linear_acceleration.x;
            dy = w1 * dy + w2 * imu_buf[ii]->linear_acceleration.y;
            dz = w1 * dz + w2 * imu_buf[ii]->linear_acceleration.z;

            rx = w1 * rx + w2 * imu_buf[ii]->angular_velocity.x;
            ry = w1 * ry + w2 * imu_buf[ii]->angular_velocity.y;
            rz = w1 * rz + w2 * imu_buf[ii]->angular_velocity.z;

            if(dx > 15.0) dx = 15.0;
            if(dy > 15.0) dy = 15.0;
            if(dz > 18.0) dz = 18.0;

            if(dx < -15.0) dx = -15.0;
            if(dy < -15.0) dy = -15.0;
            if(dz < -18.0) dz = -18.0;

            Eigen::Vector3d a1(dx, dy, dz);
            Eigen::Vector3d gy1(rx, ry, rz);

            Eigen::Vector3d un_acc_0 = Rtmp * (a0 - Batmp) - g;
            Eigen::Vector3d un_gyr = 0.5 * (gy0 + gy1) - Bgtmp;
            Rtmp *= deltaQ(un_gyr * dt1).toRotationMatrix();
            Eigen::Vector3d un_acc_1 = Rtmp * (a1 - Batmp) - g;
            Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
            Ptmp += dt1 * Vtmp + 0.5 * dt1 * dt1 * un_acc;
            Vtmp += dt1 * un_acc;

            /// 到这里，得到滑窗最旧关键帧的位姿

            vector<double*> paraBetweenEachFrame;
            // keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]: 滑窗最旧关键帧的前一关键帧对应id
            // keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width] : 滑窗最旧关键帧对应id
            // numPara: 得到期间的可以构造的约束数量
            int numPara = keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width] - keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1];
            double dQuat[numPara][4];
            double dTrans[numPara][3];

            // ||: 关键帧
            //  |: 普通帧
            /// 假设[滑窗最旧的关键帧] 和 [前一关键帧] 的中间关系如下：
            ///      || - | - | - | - ||
            /// id    2   3   4   5   6
            /// 下面的for循环，可以构造出  2-3 , 3-4, 4-5 之间的相对位姿约束，然后还剩最后一个约束，在for循环外构造

            // 计算相对位姿关系
            for(int i = keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1] + 1;
                i < keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width]; i++) {
                Eigen::Vector3d tmpTrans = Eigen::Vector3d(pose_each_frame->points[i].x,
                                                           pose_each_frame->points[i].y,
                                                           pose_each_frame->points[i].z) -
                        Eigen::Vector3d(pose_each_frame->points[i-1].x,
                        pose_each_frame->points[i-1].y,
                        pose_each_frame->points[i-1].z);
                tmpTrans = Eigen::Quaterniond(pose_info_each_frame->points[i-1].qw,
                        pose_info_each_frame->points[i-1].qx,
                        pose_info_each_frame->points[i-1].qy,
                        pose_info_each_frame->points[i-1].qz).inverse() * tmpTrans;
                dTrans[i-keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1][0] = tmpTrans.x();
                dTrans[i-keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1][1] = tmpTrans.y();
                dTrans[i-keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1][2] = tmpTrans.z();
                paraBetweenEachFrame.push_back(dTrans[i-keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1]);

                Eigen::Quaterniond tmpQuat = Eigen::Quaterniond(pose_info_each_frame->points[i-1].qw,
                        pose_info_each_frame->points[i-1].qx,
                        pose_info_each_frame->points[i-1].qy,
                        pose_info_each_frame->points[i-1].qz).inverse() *
                        Eigen::Quaterniond(pose_info_each_frame->points[i].qw,
                                           pose_info_each_frame->points[i].qx,
                                           pose_info_each_frame->points[i].qy,
                                           pose_info_each_frame->points[i].qz);
                dQuat[i-keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1][0] = tmpQuat.w();
                dQuat[i-keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1][1] = tmpQuat.x();
                dQuat[i-keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1][2] = tmpQuat.y();
                dQuat[i-keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1][3] = tmpQuat.z();
                paraBetweenEachFrame.push_back(dQuat[i-keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1]);
            }
            int jj = keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width];

            Eigen::Vector3d tmpTrans = Ptmp - Eigen::Vector3d(pose_each_frame->points[jj-1].x,
                    pose_each_frame->points[jj-1].y,
                    pose_each_frame->points[jj-1].z);
            tmpTrans = Eigen::Quaterniond(pose_info_each_frame->points[jj-1].qw,
                    pose_info_each_frame->points[jj-1].qx,
                    pose_info_each_frame->points[jj-1].qy,
                    pose_info_each_frame->points[jj-1].qz).inverse() * tmpTrans;

            dTrans[numPara-1][0] = tmpTrans.x();
            dTrans[numPara-1][1] = tmpTrans.y();
            dTrans[numPara-1][2] = tmpTrans.z();
            paraBetweenEachFrame.push_back(dTrans[numPara-1]);

            Eigen::Quaterniond qtmp(Rtmp);
            Eigen::Quaterniond tmpQuat = Eigen::Quaterniond(pose_info_each_frame->points[jj-1].qw,
                    pose_info_each_frame->points[jj-1].qx,
                    pose_info_each_frame->points[jj-1].qy,
                    pose_info_each_frame->points[jj-1].qz).inverse() * qtmp;
            dQuat[numPara-1][0] = tmpQuat.w();
            dQuat[numPara-1][1] = tmpQuat.x();
            dQuat[numPara-1][2] = tmpQuat.y();
            dQuat[numPara-1][3] = tmpQuat.z();
            paraBetweenEachFrame.push_back(dQuat[numPara-1]);

            optimizeLocalGraph(paraBetweenEachFrame);
        }

        // 如果没有开启回环检测，则不使用gtsam
        if (!loop_closure_on)
            return;

        //add poses to global graph
        // 如果关键帧数量 == 滑窗width , 即第一次优化结束
        if (pose_cloud_frame->points.size() == slide_window_width) {
            // pose_cloud_frame->points.size() - slide_window_width ==0

            // 取第一帧(普通帧？ 其实就是第一个关键帧)作为先验
            gtsam::Rot3 rotation = gtsam::Rot3::Quaternion(pose_info_each_frame->points[0].qw,
                    pose_info_each_frame->points[0].qx,
                    pose_info_each_frame->points[0].qy,
                    pose_info_each_frame->points[0].qz);
            gtsam::Point3 transition = gtsam::Point3(pose_each_frame->points[0].x,
                    pose_each_frame->points[0].y,
                    pose_each_frame->points[0].z);

            // Initialization for global pose graph
            glocal_pose_graph.add(gtsam::PriorFactor<gtsam::Pose3>(0, gtsam::Pose3(rotation, transition), prior_noise));
            glocal_init_estimate.insert(0, gtsam::Pose3(rotation, transition));

            for (int i = 0; i < 7; ++i) {
                last_pose[i] = abs_poses[abs_poses.size()-slide_window_width][i];
            }
            select_pose.x = last_pose[4];
            select_pose.y = last_pose[5];
            select_pose.z = last_pose[6];
        }
        // 如果不是第一次优化
        else if(pose_cloud_frame->points.size() > slide_window_width) {
            // keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]: 滑窗最旧关键帧的前一关键帧对应id
            // keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width] : 滑窗最旧关键帧对应id (也是下一轮的最旧关键帧id)

            /// 因子图优化，滑窗之前的的帧，包括滑窗最旧关键帧
            for(int i = keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1] + 1;
                i <= keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width]; i++) {

                gtsam::Rot3 rotationLast = gtsam::Rot3::Quaternion(pose_info_each_frame->points[i-1].qw,
                        pose_info_each_frame->points[i-1].qx,
                        pose_info_each_frame->points[i-1].qy,
                        pose_info_each_frame->points[i-1].qz);
                gtsam::Point3 transitionLast = gtsam::Point3(pose_each_frame->points[i-1].x,
                        pose_each_frame->points[i-1].y,
                        pose_each_frame->points[i-1].z);

                gtsam::Rot3 rotationCur = gtsam::Rot3::Quaternion(pose_info_each_frame->points[i].qw,
                                                                  pose_info_each_frame->points[i].qx,
                                                                  pose_info_each_frame->points[i].qy,
                                                                  pose_info_each_frame->points[i].qz);
                gtsam::Point3 transitionCur = gtsam::Point3(pose_each_frame->points[i].x,
                                                            pose_each_frame->points[i].y,
                                                            pose_each_frame->points[i].z);
                gtsam::Pose3 poseFrom = gtsam::Pose3(rotationLast, transitionLast);
                gtsam::Pose3 poseTo = gtsam::Pose3(rotationCur, transitionCur);

                glocal_pose_graph.add(gtsam::BetweenFactor<gtsam::Pose3>(i - 1,
                                                                         i,
                                                                         poseFrom.between(poseTo),
                                                                         odom_noise));
                glocal_init_estimate.insert(i, poseTo);
                //std::cout<<i<<std::endl;
            }
        }

        isam->update(glocal_pose_graph, glocal_init_estimate);
        isam->update();

        glocal_pose_graph.resize(0);
        glocal_init_estimate.clear();

        if (pose_cloud_frame->points.size() > slide_window_width) {
            for (int i = 0; i < 7; ++i) {
                last_pose[i] = abs_poses[abs_poses.size()-slide_window_width][i];
            }
            select_pose.x = last_pose[4];
            select_pose.y = last_pose[5];
            select_pose.z = last_pose[6];
        }

    }

    void correctPoses() {
        if (loop_closed == true) {
            recent_edge_keyframes.clear();
            recent_surf_keyframes.clear();

            // 取所有帧的位姿
            int numPoses = glocal_estimated.size();

            vector<Eigen::Quaterniond> quaternionRel;
            vector<Eigen::Vector3d> transitionRel;

            // 取滑动窗口内：abs_poses[]矫正之前的位姿
            // 计算滑动窗口内的相对位姿关系
            // 从滑窗
            for(int i = abs_poses.size() - slide_window_width; i < abs_poses.size() - 1; i++) {
                Eigen::Quaterniond quaternionFrom(abs_poses[i][0],
                        abs_poses[i][1],
                        abs_poses[i][2],
                        abs_poses[i][3]);
                Eigen::Vector3d transitionFrom(abs_poses[i][4],
                        abs_poses[i][5],
                        abs_poses[i][6]);

                // i+1: 最大可以== abs_poses.size() - 1
                Eigen::Quaterniond quaternionTo(abs_poses[i+1][0],
                        abs_poses[i+1][1],
                        abs_poses[i+1][2],
                        abs_poses[i+1][3]);
                Eigen::Vector3d transitionTo(abs_poses[i+1][4],
                        abs_poses[i+1][5],
                        abs_poses[i+1][6]);

                quaternionRel.push_back(quaternionFrom.inverse() * quaternionTo);
                transitionRel.push_back(quaternionFrom.inverse() * (transitionTo - transitionFrom));
            }

            // 直接更新每一帧的位姿
            for (int i = 0; i < numPoses; ++i) {
                pose_each_frame->points[i].x = glocal_estimated.at<gtsam::Pose3>(i).translation().x();
                pose_each_frame->points[i].y = glocal_estimated.at<gtsam::Pose3>(i).translation().y();
                pose_each_frame->points[i].z = glocal_estimated.at<gtsam::Pose3>(i).translation().z();

                pose_info_each_frame->points[i].x = pose_each_frame->points[i].x;
                pose_info_each_frame->points[i].y = pose_each_frame->points[i].y;
                pose_info_each_frame->points[i].z = pose_each_frame->points[i].z;
                pose_info_each_frame->points[i].qw = glocal_estimated.at<gtsam::Pose3>(i).rotation().toQuaternion().w();
                pose_info_each_frame->points[i].qx = glocal_estimated.at<gtsam::Pose3>(i).rotation().toQuaternion().x();
                pose_info_each_frame->points[i].qy = glocal_estimated.at<gtsam::Pose3>(i).rotation().toQuaternion().y();
                pose_info_each_frame->points[i].qz = glocal_estimated.at<gtsam::Pose3>(i).rotation().toQuaternion().z();
            }

            // 更新abs_poses[]、Rs[]、Ps[]、pose_cloud_frame[]、pose_info_cloud_frame[]
            // 直到滑窗最旧关键帧
            for(int i = 0; i <= pose_cloud_frame->points.size() - slide_window_width; i++) {
                // 更新pose_cloud_frame[]、pose_info_cloud_frame[]
                pose_cloud_frame->points[i].x = pose_each_frame->points[keyframe_id_in_frame[i]].x;
                pose_cloud_frame->points[i].y = pose_each_frame->points[keyframe_id_in_frame[i]].y;
                pose_cloud_frame->points[i].z = pose_each_frame->points[keyframe_id_in_frame[i]].z;

                pose_info_cloud_frame->points[i].x = pose_each_frame->points[keyframe_id_in_frame[i]].x;
                pose_info_cloud_frame->points[i].y = pose_each_frame->points[keyframe_id_in_frame[i]].y;
                pose_info_cloud_frame->points[i].z = pose_each_frame->points[keyframe_id_in_frame[i]].z;
                pose_info_cloud_frame->points[i].qw = pose_info_each_frame->points[keyframe_id_in_frame[i]].qw;
                pose_info_cloud_frame->points[i].qx = pose_info_each_frame->points[keyframe_id_in_frame[i]].qx;
                pose_info_cloud_frame->points[i].qy = pose_info_each_frame->points[keyframe_id_in_frame[i]].qy;
                pose_info_cloud_frame->points[i].qz = pose_info_each_frame->points[keyframe_id_in_frame[i]].qz;

                // 更新abs_poses[]、Rs[]、Ps[]
                abs_poses[i+1][0] = pose_info_cloud_frame->points[i].qw;
                abs_poses[i+1][1] = pose_info_cloud_frame->points[i].qx;
                abs_poses[i+1][2] = pose_info_cloud_frame->points[i].qy;
                abs_poses[i+1][3] = pose_info_cloud_frame->points[i].qz;
                abs_poses[i+1][4] = pose_info_cloud_frame->points[i].x;
                abs_poses[i+1][5] = pose_info_cloud_frame->points[i].y;
                abs_poses[i+1][6] = pose_info_cloud_frame->points[i].z;

                Rs[i+1] = Eigen::Quaterniond(abs_poses[i+1][0],
                        abs_poses[i+1][1],
                        abs_poses[i+1][2],
                        abs_poses[i+1][3]).toRotationMatrix();

                Ps[i+1][0] = abs_poses[i+1][4];
                Ps[i+1][1] = abs_poses[i+1][5];
                Ps[i+1][2] = abs_poses[i+1][6];
            }

            // 更新传播，因为gtsam优化只优化到 滑窗之前的帧（包括滑窗最旧关键帧），需要将矫正信息传播到滑窗的帧
            // 从滑窗最旧帧开始更新
            // 先更新滑窗最旧帧的后一关键帧
            for(int i = abs_poses.size() - slide_window_width; i < abs_poses.size() - 1; i++) {
                // 取前一关键帧位姿
                Eigen::Quaterniond integratedQuaternion(abs_poses[i][0],
                        abs_poses[i][1],
                        abs_poses[i][2],
                        abs_poses[i][3]);
                Eigen::Vector3d integratedTransition(abs_poses[i][4],
                        abs_poses[i][5],
                        abs_poses[i][6]);

                // transitionRel[] , quaternionRel[]: 当前帧到前一帧的变换
                //
                integratedTransition = integratedTransition + integratedQuaternion * transitionRel[i - abs_poses.size() + slide_window_width];
                integratedQuaternion = integratedQuaternion * quaternionRel[i - abs_poses.size() + slide_window_width];

                //
                abs_poses[i+1][0] = integratedQuaternion.w();
                abs_poses[i+1][1] = integratedQuaternion.x();
                abs_poses[i+1][2] = integratedQuaternion.y();
                abs_poses[i+1][3] = integratedQuaternion.z();
                abs_poses[i+1][4] = integratedTransition.x();
                abs_poses[i+1][5] = integratedTransition.y();
                abs_poses[i+1][6] = integratedTransition.z();

                Rs[i+1] = Eigen::Quaterniond(abs_poses[i+1][0],
                        abs_poses[i+1][1],
                        abs_poses[i+1][2],
                        abs_poses[i+1][3]).toRotationMatrix();

                Ps[i+1][0] = abs_poses[i+1][4];
                Ps[i+1][1] = abs_poses[i+1][5];
                Ps[i+1][2] = abs_poses[i+1][6];

                pose_cloud_frame->points[i].x = abs_poses[i+1][4];
                pose_cloud_frame->points[i].y = abs_poses[i+1][5];
                pose_cloud_frame->points[i].z = abs_poses[i+1][6];

                pose_info_cloud_frame->points[i].x = abs_poses[i+1][4];
                pose_info_cloud_frame->points[i].y = abs_poses[i+1][5];
                pose_info_cloud_frame->points[i].z = abs_poses[i+1][6];
                pose_info_cloud_frame->points[i].qw = abs_poses[i+1][0];
                pose_info_cloud_frame->points[i].qx = abs_poses[i+1][1];
                pose_info_cloud_frame->points[i].qy = abs_poses[i+1][2];
                pose_info_cloud_frame->points[i].qz = abs_poses[i+1][3];
            }

            // 取矫正后的最新关键帧
            abs_pose = abs_poses.back();
            // 取滑窗最旧关键帧作为last_pose
            for (int i = 0; i < 7; ++i) {
                last_pose[i] = abs_poses[abs_poses.size() - slide_window_width][i];
            }

            // 取滑窗最旧关键帧作为回环检测的source
            select_pose.x = last_pose[4];
            select_pose.y = last_pose[5];
            select_pose.z = last_pose[6];

            loop_closed = false;
            // 由于矫正了，所以之前的marg信息去掉
            marg = false;
        }
    }

    void publishOdometry() {
        // 关键帧数量足够滑窗优化
        if(pose_info_cloud_frame->points.size() >= slide_window_width) {
            odom_mapping.header.stamp = ros::Time().fromSec(time_new_odom);
            odom_mapping.pose.pose.orientation.w = pose_info_cloud_frame->points[pose_info_cloud_frame->points.size()-slide_window_width].qw;
            odom_mapping.pose.pose.orientation.x = pose_info_cloud_frame->points[pose_info_cloud_frame->points.size()-slide_window_width].qx;
            odom_mapping.pose.pose.orientation.y = pose_info_cloud_frame->points[pose_info_cloud_frame->points.size()-slide_window_width].qy;
            odom_mapping.pose.pose.orientation.z = pose_info_cloud_frame->points[pose_info_cloud_frame->points.size()-slide_window_width].qz;
            odom_mapping.pose.pose.position.x = pose_info_cloud_frame->points[pose_info_cloud_frame->points.size()-slide_window_width].x;
            odom_mapping.pose.pose.position.y = pose_info_cloud_frame->points[pose_info_cloud_frame->points.size()-slide_window_width].y;
            odom_mapping.pose.pose.position.z = pose_info_cloud_frame->points[pose_info_cloud_frame->points.size()-slide_window_width].z;
            pub_odom.publish(odom_mapping);
        }

        sensor_msgs::PointCloud2 msgs;

        if (pub_poses.getNumSubscribers() && pose_info_cloud_frame->points.size() >= slide_window_width) {
            pcl::toROSMsg(*pose_each_frame, msgs);
            msgs.header.stamp = ros::Time().fromSec(time_new_odom);
            msgs.header.frame_id = frame_id;
            pub_poses.publish(msgs);
        }


        PointPoseInfo Tbl;
        Tbl.qw = q_bl.w();
        Tbl.qx = q_bl.x();
        Tbl.qy = q_bl.y();
        Tbl.qz = q_bl.z();
        Tbl.x = t_bl.x();
        Tbl.y = t_bl.y();
        Tbl.z = t_bl.z();

        // publish the corner and surf feature points in lidar_init frame
        if (pub_edge.getNumSubscribers()) {
            for (int i = 0; i < edge_last_ds->points.size(); ++i) {
                transformPoint(&edge_last_ds->points[i], &edge_last_ds->points[i], q_bl, t_bl);
                transformPoint(&edge_last_ds->points[i], &edge_last_ds->points[i]);
            }
            pcl::toROSMsg(*edge_last_ds, msgs);
            msgs.header.stamp = ros::Time().fromSec(time_new_odom);
            msgs.header.frame_id = frame_id;
            pub_edge.publish(msgs);
        }

        if (pub_surf.getNumSubscribers()) {
            for (int i = 0; i < surf_last_ds->points.size(); ++i) {
                transformPoint(&surf_last_ds->points[i], &surf_last_ds->points[i], q_bl, t_bl);
                transformPoint(&surf_last_ds->points[i], &surf_last_ds->points[i]);
            }
            pcl::toROSMsg(*surf_last_ds, msgs);
            msgs.header.stamp = ros::Time().fromSec(time_new_odom);
            msgs.header.frame_id = frame_id;
            pub_surf.publish(msgs);
        }

        if (pub_full.getNumSubscribers()) {
            for (int i = 0; i < full_cloud->points.size(); ++i) {
                transformPoint(&full_cloud->points[i], &full_cloud->points[i], q_bl, t_bl);
                transformPoint(&full_cloud->points[i], &full_cloud->points[i]);
            }
            pcl::toROSMsg(*full_cloud, msgs);
            msgs.header.stamp = ros::Time().fromSec(time_new_odom);
            msgs.header.frame_id = frame_id;
            pub_full.publish(msgs);
        }
    }

    void clearCloud() {
        edge_local_map->clear();
        edge_local_map_ds->clear();
        surf_local_map->clear();
        surf_local_map_ds->clear();

        if(surf_lasts_ds.size() > slide_window_width + 5) {
            surf_lasts_ds[surf_lasts_ds.size() - slide_window_width - 6]->clear();
        }

        if(pre_integrations.size() > slide_window_width + 5) {
            pre_integrations[pre_integrations.size() - slide_window_width - 6] = nullptr;
        }

        if(last_marginalization_parameter_blocks.size() > slide_window_width + 5) {
            last_marginalization_parameter_blocks[last_marginalization_parameter_blocks.size() - slide_window_width - 6] = nullptr;
        }
    }

    void loopClosureThread() {
        if (!loop_closure_on)
            return;

        ros::Rate rate(1);
        while (ros::ok()) {
            rate.sleep();
            performLoopClosure();
        }
    }

    bool detectLoopClosure() {
        latest_key_frames->clear();
        latest_key_frames_ds->clear();
        his_key_frames->clear();
        his_key_frames_ds->clear();

        std::lock_guard<std::mutex> lock(mutual_exclusion);

        // Look for the closest key frames
        std::vector<int> pt_search_idxLoop;
        std::vector<float> pt_search_sq_distsLoop;

        kd_tree_his_key_poses->setInputCloud(pose_cloud_frame);
        kd_tree_his_key_poses->radiusSearch(select_pose, lc_search_radius, pt_search_idxLoop, pt_search_sq_distsLoop, 0);

        closest_his_idx = -1;
        for (int i = 0; i < pt_search_idxLoop.size(); ++i) {
            int idx = pt_search_idxLoop[i];
            if (abs(pose_info_cloud_frame->points[idx].time - time_new_odom) > lc_time_thres) {
                closest_his_idx = idx;
                break;
            }
        }

        if (closest_his_idx == -1)
            return false;
        else if(abs(time_last_loop - time_new_odom) < 0.2)
            return false;

        //ROS_INFO("******************* Loop closure ready to detect! *******************");

        // Combine the corner and surf frames to form the latest frame
        latest_frame_idx_loop = pose_cloud_frame->points.size() - slide_window_width;

        // 把滑窗最旧帧附近的几帧作为 source
        for (int j = 0; j < 6; ++j) {
            if (latest_frame_idx_loop-j < 0)
                continue;
            Eigen::Quaterniond q_po(pose_info_cloud_frame->points[latest_frame_idx_loop-j].qw,
                    pose_info_cloud_frame->points[latest_frame_idx_loop-j].qx,
                    pose_info_cloud_frame->points[latest_frame_idx_loop-j].qy,
                    pose_info_cloud_frame->points[latest_frame_idx_loop-j].qz);

            Eigen::Vector3d t_po(pose_info_cloud_frame->points[latest_frame_idx_loop-j].x,
                    pose_info_cloud_frame->points[latest_frame_idx_loop-j].y,
                    pose_info_cloud_frame->points[latest_frame_idx_loop-j].z);

            Eigen::Quaterniond q_tmp = q_po * q_bl;
            Eigen::Vector3d t_tmp = q_po * t_bl + t_po;

            *latest_key_frames += *transformCloud(edge_frames[latest_frame_idx_loop-j], q_tmp, t_tmp);
            *latest_key_frames += *transformCloud(surf_frames[latest_frame_idx_loop-j], q_tmp, t_tmp);
        }

        ds_filter_his_frames.setInputCloud(latest_key_frames);
        ds_filter_his_frames.filter(*latest_key_frames_ds);


        // Form the history frame for loop closure detection
        // 取回环候选帧附近的帧构建局部地图，作为回环的target
        for (int j = -lc_map_width; j <= lc_map_width; ++j) {
            if (closest_his_idx + j < 0 || closest_his_idx + j > latest_frame_idx_loop)
                continue;

            Eigen::Quaterniond q_po(pose_info_cloud_frame->points[closest_his_idx+j].qw,
                    pose_info_cloud_frame->points[closest_his_idx+j].qx,
                    pose_info_cloud_frame->points[closest_his_idx+j].qy,
                    pose_info_cloud_frame->points[closest_his_idx+j].qz);

            Eigen::Vector3d t_po(pose_info_cloud_frame->points[closest_his_idx+j].x,
                    pose_info_cloud_frame->points[closest_his_idx+j].y,
                    pose_info_cloud_frame->points[closest_his_idx+j].z);

            Eigen::Quaterniond q_tmp = q_po * q_bl;
            Eigen::Vector3d t_tmp = q_po * t_bl + t_po;

            *his_key_frames += *transformCloud(edge_frames[closest_his_idx+j], q_tmp, t_tmp);
            *his_key_frames += *transformCloud(surf_frames[closest_his_idx+j], q_tmp, t_tmp);
        }

        ds_filter_his_frames.setInputCloud(his_key_frames);
        ds_filter_his_frames.filter(*his_key_frames_ds);

        return true;
    }

    void performLoopClosure() {
        if (pose_cloud_frame->points.empty())
            return;

        if (!loop_to_close) {
            if (detectLoopClosure())
                loop_to_close = true;
            if (!loop_to_close)
                return;
        }

        loop_to_close = false;

        pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(30);
        icp.setMaximumIterations(100);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(5);

        // 滑窗内的关键帧构建的source 点云
        icp.setInputSource(latest_key_frames_ds);
        // 闭环候选帧附近构建的局部地图
        icp.setInputTarget(his_key_frames_ds);
        pcl::PointCloud<PointType>::Ptr alignedCloud(new pcl::PointCloud<PointType>());
        icp.align(*alignedCloud);

        //std::cout << "ICP converg flag:" << icp.hasConverged() << ". Fitness score: " << icp.getFitnessScore() << endl;

        if (!icp.hasConverged() || icp.getFitnessScore() > lc_icp_thres)
            return;

        Timer t_loop("Loop Closure");
        //ROS_INFO("******************* Loop closure detected! *******************");

        // icp结果: 当前点云地图坐标系 到 原始点云地图坐标系的变换
        Eigen::Matrix4d correctedTranform;
        correctedTranform = icp.getFinalTransformation().cast<double>();
        Eigen::Quaterniond quaternionIncre(correctedTranform.block<3, 3>(0, 0));
        Eigen::Vector3d transitionIncre(correctedTranform.block<3, 1>(0, 3));
        Eigen::Quaterniond quaternionToCorrect(pose_info_cloud_frame->points[latest_frame_idx_loop].qw,
                                               pose_info_cloud_frame->points[latest_frame_idx_loop].qx,
                                               pose_info_cloud_frame->points[latest_frame_idx_loop].qy,
                                               pose_info_cloud_frame->points[latest_frame_idx_loop].qz);
        Eigen::Vector3d transitionToCorrect(pose_info_cloud_frame->points[latest_frame_idx_loop].x,
                                            pose_info_cloud_frame->points[latest_frame_idx_loop].y,
                                            pose_info_cloud_frame->points[latest_frame_idx_loop].z);

        // 得到：矫正后的滑窗最旧关键帧的位姿
        Eigen::Quaterniond quaternionCorrected = quaternionIncre * quaternionToCorrect;
        Eigen::Vector3d transitionCorrected = quaternionIncre * transitionToCorrect + transitionIncre;

        gtsam::Rot3 rotationFrom = gtsam::Rot3::Quaternion(quaternionCorrected.w(), quaternionCorrected.x(), quaternionCorrected.y(), quaternionCorrected.z());
        gtsam::Point3 transitionFrom = gtsam::Point3(transitionCorrected.x(), transitionCorrected.y(), transitionCorrected.z());

        gtsam::Rot3 rotationTo = gtsam::Rot3::Quaternion(pose_info_cloud_frame->points[closest_his_idx].qw,
                                                         pose_info_cloud_frame->points[closest_his_idx].qx,
                                                         pose_info_cloud_frame->points[closest_his_idx].qy,
                                                         pose_info_cloud_frame->points[closest_his_idx].qz);
        gtsam::Point3 transitionTo = gtsam::Point3(pose_info_cloud_frame->points[closest_his_idx].x,
                                                   pose_info_cloud_frame->points[closest_his_idx].y,
                                                   pose_info_cloud_frame->points[closest_his_idx].z);

        gtsam::Pose3 poseFrom = gtsam::Pose3(rotationFrom, transitionFrom);
        gtsam::Pose3 poseTo = gtsam::Pose3(rotationTo, transitionTo);
        gtsam::Vector vector6(6);
        double noiseScore = icp.getFitnessScore();
        vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
        constraint_noise = gtsam::noiseModel::Diagonal::Variances(vector6);

        std::lock_guard<std::mutex> lock(mutual_exclusion);

        // 加入回环因子： 滑窗最旧关键帧 到 闭环关键帧的 相对位姿关系
        glocal_pose_graph.add(gtsam::BetweenFactor<gtsam::Pose3>(keyframe_id_in_frame[latest_frame_idx_loop],
                                                                 keyframe_id_in_frame[closest_his_idx],
                                                                 poseFrom.between(poseTo),
                                                                 constraint_noise));
        isam->update(glocal_pose_graph);
        isam->update();
        glocal_pose_graph.resize(0);

        loop_closed = true;

        glocal_estimated = isam->calculateEstimate();
        correctPoses();

        // 清空marg信息
        if (last_marginalization_info) {
            delete last_marginalization_info;
        }
        last_marginalization_info = nullptr;

        time_last_loop = pose_info_cloud_frame->points[latest_frame_idx_loop].time;

        //ROS_INFO("******************* Loop closure finished! *******************");
        //t_loop.tic_toc();
    }

    void publishCompleteMap() {
        if (pose_cloud_frame->points.size() > 10) {
            for (int i = 0; i < pose_info_cloud_frame->points.size(); i = i + mapping_interval) {
                Eigen::Quaterniond q_po(pose_info_cloud_frame->points[i].qw,
                                        pose_info_cloud_frame->points[i].qx,
                                        pose_info_cloud_frame->points[i].qy,
                                        pose_info_cloud_frame->points[i].qz);

                Eigen::Vector3d t_po(pose_info_cloud_frame->points[i].x,
                                     pose_info_cloud_frame->points[i].y,
                                     pose_info_cloud_frame->points[i].z);

                Eigen::Quaterniond q_tmp = q_po * q_bl;
                Eigen::Vector3d t_tmp = q_po * t_bl + t_po;

                PointPoseInfo Ttmp;
                Ttmp.qw = q_tmp.w();
                Ttmp.qx = q_tmp.x();
                Ttmp.qy = q_tmp.y();
                Ttmp.qz = q_tmp.z();
                Ttmp.x = t_tmp.x();
                Ttmp.y = t_tmp.y();
                Ttmp.z = t_tmp.z();

                *global_map += *transformCloud(full_clouds_ds[i], &Ttmp);
            }

            ds_filter_global_map.setInputCloud(global_map);
            ds_filter_global_map.filter(*global_map_ds);

            sensor_msgs::PointCloud2 msgs;
            pcl::toROSMsg(*global_map, msgs);
            msgs.header.stamp = ros::Time().fromSec(time_new_odom);
            msgs.header.frame_id = frame_id;
            pub_map.publish(msgs);
            global_map->clear();
            global_map_ds->clear();
        }
    }

    void mapVisualizationThread() {
        ros::Rate rate(0.038);
        while (ros::ok()) {
            rate.sleep();
            //ROS_INFO("Publishing the map");
            publishCompleteMap();
        }

        if(!save_pcd)
            return;

        cout << "****************************************************" << endl;
        cout << "Saving map to pcd files ..." << endl;

        PointPoseInfo Tbl;
        Tbl.qw = q_bl.w();
        Tbl.qx = q_bl.x();
        Tbl.qy = q_bl.y();
        Tbl.qz = q_bl.z();
        Tbl.x = t_bl.x();
        Tbl.y = t_bl.y();
        Tbl.z = t_bl.z();

        for (int i = 0; i < pose_info_cloud_frame->points.size(); i = i + mapping_interval) {
            *global_map += *transformCloud(transformCloud(surf_frames[i], &Tbl), &pose_info_cloud_frame->points[i]);
        }
        ds_filter_global_map.setInputCloud(global_map);
        ds_filter_global_map.filter(*global_map_ds);
        pcl::io::savePCDFileASCII("/home/mli/MengLi/pcd/global_map.pcd", *global_map_ds);
        cout << "****************************************************" << endl;
        cout << "Saving map to pcd files completed" << endl;
        global_map->clear();
        global_map_ds->clear();
    }

    void run() {
        // 数据同步这么来？
        if (new_surf && new_edge && new_odom && new_each_odom && new_full_cloud) {
            new_edge = false;
            new_surf = false;
            new_odom = false;
            new_each_odom = false;
            new_full_cloud = false;

            std::lock_guard<std::mutex> lock(mutual_exclusion);

            //cout<<"map_pub_cnt: "<<++map_pub_cnt<<endl;

            Timer t_map("BackendFusion");
            buildLocalMapWithLandMark();
            downSampleCloud();
            saveKeyFramesAndFactors();
            publishOdometry();
            clearCloud();
            //t_map.tic_toc();
            runtime += t_map.toc();
            //cout<<"BackendFusion average run time: "<<runtime / each_odom_buf.size()<<endl;
        }
    }
};

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);

    ros::init(argc, argv, "lili_om_rot");

    ROS_INFO("\033[1;32m---->\033[0m Back End Started.");

    BackendFusion BEF;

    // 启动两个线程
    std::thread threadLoopClosure(&BackendFusion::loopClosureThread, &BEF);
    std::thread threadMapVisualization(&BackendFusion::mapVisualizationThread, &BEF);

    ros::Rate rate(200);

    while (ros::ok()) {
        ros::spinOnce();
        // 主线
        BEF.run();

        rate.sleep();
    }

    threadLoopClosure.join();
    threadMapVisualization.join();

    return 0;
}
