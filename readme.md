# About

This is for the PPCA 2018 Deep Learning System Test

# How To Run

For the first use, please install Intel® Math Kernel Library by run this command:
```bash
sudo sh mkl_install.sh
```

For the test, please enter `./dlsystem_test`, add environment variables and make the library:
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_LIB:/opt/intel/lib/intel64:/opt/intel/mkl/lib/intel64
make
```
Then run test by use the command:
```bash
python3 run_test.py TensorFreeze
```