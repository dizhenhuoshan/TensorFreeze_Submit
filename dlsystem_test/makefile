TARGET=boost_cc.so
SRCS = boost_cc.cpp

OBJS = $(SRCS:.c=.o)

$(TARGET):
	g++ -o boost_cc.so boost_cc.cpp -O4 -shared -fPIC -std=c++11 -I/opt/intel/mkl/include -L/opt/intel/mkl/lib/intel64 -L/opt/intel/lib/intel64 -lmkl_rt -lmkl_gf_lp64 -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_mc3 -lmkl_avx2 -lmkl_core -march=corei7-avx

clean:
	rm -rf boost_cc.so
