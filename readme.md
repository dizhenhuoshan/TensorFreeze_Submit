#About
This is for the PPCA 2018 Deep Learning System Test

#How To Run
For the first use, please install Intel® Math Kernel Library by run this command:
```bash
sudo sh mkl_install.sh
```

For the test, please enter `./dlsystem_test` and make the dynamic link library:
```bash
make
```
Then run test by use the command:
```bash
python3 run_test.py TensorFreeze
```