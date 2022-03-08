// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include <iostream>
#include <memory>
#include <thread>
#include "Eigen/Dense"
#include <string>

#include "Open3DNew.h"
#include "nanoflann.hpp"
#include "TestVisualizer.h"


using namespace std;
//--------------------

class Geometry {
public:
    /// \enum GeometryType
    /// \brief Specifies possible geometry types.
    enum class GeometryType {
        /// Unspecified geometry type.
        Unspecified = 0,
        /// PointCloud
        PointCloud = 1,

    };

public:
    virtual ~Geometry() {}

protected:
    /// \brief Parameterized Constructor.
    /// \param type Specifies the type of geometry of the object constructed.
    /// \param dimension Specifies whether the dimension is 2D or 3D.
    Geometry(GeometryType type, int dimension)
        : geometry_type_(type), dimension_(dimension) {}

public:
    /// Clear all elements in the geometry.
    virtual Geometry &Clear() = 0;
    /// Returns `true` iff the geometry is empty.
    virtual bool IsEmpty() const = 0;
    /// Returns one of registered geometry types.
    GeometryType GetGeometryType() const { return geometry_type_; }
    /// Returns whether the geometry is 2D or 3D.
    int Dimension() const { return dimension_; }

    std::string GetName() const { return name_; }
    void SetName(const std::string &name) { name_ = name; }

private:
    /// Type of geometry from GeometryType.
    GeometryType geometry_type_ = GeometryType::Unspecified;
    /// Number of dimensions of the geometry.
    int dimension_ = 3;
    std::string name_;
};

// Reversed engineered and simplified class, originally from
// open3d::geometry::KDtreeFlann
class KNearestNeighbour {
public:
    //    KNearestNeighbour(const open3d::pipelines::registration::Feature
    //    &feature) {
    KNearestNeighbour(const open3d::pipelines::registration::Feature &feature) {
        SetFeature(feature);
    }
    //   KNearestNeighbour(){}
    bool SetMatrixData(const Eigen::MatrixXd &data) {
        return SetRawData(Eigen::Map<const Eigen::MatrixXd>(
                data.data(), data.rows(), data.cols()));
    }

    bool SetGeometry(const Geometry &geometry) {
        return SetRawData(Eigen::Map<const Eigen::MatrixXd>(
                (const double *)((const open3d::geometry::PointCloud &)geometry)
                        .points_.data(),
                3,
                ((const open3d::geometry::PointCloud &)geometry)
                        .points_.size()));
    }

    bool SetFeature(const open3d::pipelines::registration::Feature &feature) {
        return SetMatrixData(feature.data_);
    }

    bool SetRawData(const Eigen::Map<const Eigen::MatrixXd> &data) {
        dimension_ = data.rows();
        dataset_size_ = data.cols();
        if (dimension_ == 0 || dataset_size_ == 0) {
            open3d::utility::LogWarning(
                    "[KDTreeFlann::SetRawData] Failed due to no data.");
            return false;
        }
        data_.resize(dataset_size_ * dimension_);
        memcpy(data_.data(), data.data(),
               dataset_size_ * dimension_ * sizeof(double));
        data_interface_.reset(new Eigen::Map<const Eigen::MatrixXd>(data));
        nanoflann_index_.reset(
                new KDTree_t(dimension_, std::cref(*data_interface_), 15));
        nanoflann_index_->index->buildIndex();
        return true;
    }

    // public:
    template <typename T>
    int SearchKNN(const T &query,
                  int knn,
                  std::vector<int> &indices,
                  std::vector<double> &distance2) const {
        // This is optimized code for heavily repeated search.
        // Other flann::Index::knnSearch() implementations lose performance due
        // to memory allocation/deallocation.
        if (data_.empty() || dataset_size_ <= 0 ||
            size_t(query.rows()) != dimension_ || knn < 0) {
            return -1;
        }
        indices.resize(knn);
        distance2.resize(knn);
        std::vector<Eigen::Index> indices_eigen(knn);
        int k = nanoflann_index_->index->knnSearch(
                query.data(), knn, indices_eigen.data(), distance2.data());
        indices.resize(k);
        distance2.resize(k);
        std::copy_n(indices_eigen.begin(), k, indices.begin());
        return k;
    }

    template <typename T>
    int Search(const T &query,
               const KDTreeSearchParam &param,
               std::vector<int> &indices,
               std::vector<double> &distance2) const {
        switch (param.GetSearchType()) {
            case KDTreeSearchParam::SearchType::Knn:
                return SearchKNN(query,
                                 ((const KDTreeSearchParamKNN &)param).knn_,
                                 indices, distance2);
            case KDTreeSearchParam::SearchType::Radius:
                return SearchRadius(
                        query, ((const KDTreeSearchParamRadius &)param).radius_,
                        indices, distance2);
            case KDTreeSearchParam::SearchType::Hybrid:
                return SearchHybrid(
                        query, ((const KDTreeSearchParamHybrid &)param).radius_,
                        ((const KDTreeSearchParamHybrid &)param).max_nn_,
                        indices, distance2);
            default:
                return -1;
        }
        return -1;
    }

protected:
    using KDTree_t = nanoflann::KDTreeEigenMatrixAdaptor<
            Eigen::Map<const Eigen::MatrixXd>,
            -1,
            nanoflann::metric_L2,
            false>;

    std::vector<double> data_;
    std::unique_ptr<Eigen::Map<const Eigen::MatrixXd>> data_interface_;
    std::unique_ptr<KDTree_t> nanoflann_index_;
    size_t dimension_ = 0;
    size_t dataset_size_ = 0;
};

// Function to pre-process pointcloud
std::tuple<std::shared_ptr<open3d::geometry::PointCloud>,
           std::shared_ptr<open3d::pipelines::registration::Feature>>
PreprocessPointCloud(const char *file_name) {
    auto pcd = open3d::io::CreatePointCloudFromFile(file_name);
    auto pcd_down = pcd->VoxelDownSample(0.05);
    pcd_down->EstimateNormals(
            open3d::geometry::KDTreeSearchParamHybrid(0.1, 30));
    auto pcd_fpfh = open3d::pipelines::registration::ComputeFPFHFeature(
            *pcd_down, open3d::geometry::KDTreeSearchParamHybrid(0.25, 100));
    return std::make_tuple(pcd_down, pcd_fpfh);
}

// function to visualise registration
void VisualizeRegistration(const open3d::geometry::PointCloud &source,
                           const open3d::geometry::PointCloud &target,
                           const Eigen::Matrix4d &Transformation) {
    std::shared_ptr<open3d::geometry::PointCloud> source_transformed_ptr(
            new open3d::geometry::PointCloud);
    std::shared_ptr<open3d::geometry::PointCloud> target_ptr(
            new open3d::geometry::PointCloud);
    *source_transformed_ptr = source;
    *target_ptr = target;
    source_transformed_ptr->Transform(Transformation);
    open3d::visualization::DrawGeometries({source_transformed_ptr, target_ptr},
                                          "Registration result");
}

void icpdata(const char* data, const char* dataSource)
{
    using namespace open3d;
    //declare and pre-process data (lower down to x voxel)
    std::shared_ptr<geometry::PointCloud> source, target;
    std::shared_ptr<pipelines::registration::Feature> source_fpfh, target_fpfh;
    std::tie(source, source_fpfh) = PreprocessPointCloud(dataSource);
    std::tie(target, target_fpfh) = PreprocessPointCloud(data);

    // Prepare checkers
    std::vector<std::reference_wrapper<
        const pipelines::registration::CorrespondenceChecker>>
        correspondence_checker;
    auto correspondence_checker_edge_length =
        pipelines::registration::CorrespondenceCheckerBasedOnEdgeLength(
            0.9);
    auto correspondence_checker_distance =
        pipelines::registration::CorrespondenceCheckerBasedOnDistance(
            0.075);
    auto correspondence_checker_normal =
        pipelines::registration::CorrespondenceCheckerBasedOnNormal(
            0.52359878);
    correspondence_checker.push_back(correspondence_checker_edge_length);
    correspondence_checker.push_back(correspondence_checker_distance);
    correspondence_checker.push_back(correspondence_checker_normal);



    // Find correspondence filter by mutual filter
    int nPti = int(source->points_.size());
    int nPtj = int(target->points_.size());

    KNearestNeighbour feature_tree_i(*source_fpfh);
    KNearestNeighbour feature_tree_j(*target_fpfh);

    std::vector<Eigen::Vector2i> corres_ji;

    std::vector<int> i_to_j(nPti, -1);

    // Buffer all correspondences
    for (int j = 0; j < nPtj; j++) {
        std::vector<int> corres_tmp(1);
        std::vector<double> dist_tmp(1);

        feature_tree_i.SearchKNN(Eigen::VectorXd(target_fpfh->data_.col(j)),
            1, corres_tmp, dist_tmp);
        int i = corres_tmp[0];
        corres_ji.push_back(Eigen::Vector2i(i, j));
    }

    //set mutual filter
    std::vector<Eigen::Vector2i> mutual;

    for (auto& corres : corres_ji) {
        int j = corres(1);
        int j2i = corres(0);

        std::vector<int> corres_tmp(1);
        std::vector<double> dist_tmp(1);
        feature_tree_j.SearchKNN(
            Eigen::VectorXd(source_fpfh->data_.col(j2i)), 1,
            corres_tmp, dist_tmp);

        int i2j = corres_tmp[0];
        if (i2j == j) {
            mutual.push_back(corres);
        }
    }

    utility::LogDebug("{:d} points remain", mutual.size());

    //perform fast global registration

    Eigen::Matrix4d resultFastGlobalTrans = Eigen::Matrix4d::Identity();


    auto resultFGT = pipelines::registration::FastGlobalRegistrationBasedOnCorrespondence(*source, *target, mutual);
    resultFastGlobalTrans = resultFGT.transformation_;

    //------ICP Fine tuning+++++++++++++++---------


    std::vector<int> iterations = { 50, 30, 14 };
    Eigen::Matrix4d trans = Eigen::Matrix4d::Identity();

    auto result = pipelines::registration::RegistrationGeneralizedICP(
        *source, *target, 0.07, resultFastGlobalTrans,
        pipelines::registration::TransformationEstimationForGeneralizedICP(),
        pipelines::registration::ICPConvergenceCriteria(1e-6, 1e-6, iterations[3]));

    //standard ICP
    trans = result.transformation_;

    //visualise the transformation
    VisualizeRegistration(*source, *target, trans);
    VisualizeRegistration(*source, *target, resultFastGlobalTrans);

    //----------+++++++++++++++++++------------

    std::stringstream ss;
    ss << trans;
    utility::LogInfo("Final transformation = \n{}", ss.str());

}

//int main(int argc, char *argv[]) {
//    
//    using namespace open3d;
//    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);
//
//    //LOAD DATA-----------
//    
//    auto cloud_ptr_target = std::make_shared<geometry::PointCloud>();
//    auto cloud_ptr_source = std::make_shared<geometry::PointCloud>();
//    
//    char *data = "C:/Users/afiqa/Open3D/examples/test_data/ICP/cloud_bin_1.pcd";
//    char *dataSource = "C:/Users/afiqa//Open3D/examples/test_data/ICP/cloud_bin_0.pcd";
//
//    icpdata(data, dataSource);
//    //------------------ Fast Global Transform++++++++++++++
//    
//  
// 
//    return 0;
//    
//}
