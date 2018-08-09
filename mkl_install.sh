wget http://registrationcenter-download.intel.com/akdlm/irc_nas/tec/13005/l_mkl_2018.3.222.tgz
tar -zxvf l_mkl_2018.3.222.tgz
cd ./l_mkl_2018.3.222
sudo sh install.sh
cd /opt/intel/compilers_and_libraries_2018.3.222/linux/mkl/bin
sudo sh mklvars.sh intel64
