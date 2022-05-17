#include "toyslam/evaluate_trajectory.h"
#include "toyslam/config.h"

using namespace toyslam;
using namespace std;

int main () 
{
  std::string config_file_path = "/home/tianze/toySLAM/config/default.yaml";
  Config::SetParameterFile(config_file_path);
  std::string groundtruth_file = Config::Get<std::string>("groundtruth_file_path");
  std::string estimated_file = Config::Get<std::string>("outfile_path");
  cout << "Evaluating estimation in " << estimated_file
       << " against" << groundtruth_file << endl;
  // Evaluate the trajectory.
  evaluate(groundtruth_file, estimated_file);

  return 0;
}