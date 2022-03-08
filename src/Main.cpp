
#include "Open3DNew.h"
#include "TestVisualizer.h"

            int main(int argc, char* argv[]) {

                using namespace open3d;
                utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);


                auto cloud_ptr_target = std::make_shared<geometry::PointCloud>();
                auto cloud_ptr_source = std::make_shared<geometry::PointCloud>();

                char* data = "C:/Users/afiqa/Open3D/examples/test_data/ICP/cloud_bin_1.pcd";
                char* dataSource = "C:/Users/afiqa//Open3D/examples/test_data/ICP/cloud_bin_0.pcd";

                icpdata(data, dataSource);

                return 0;

            }
