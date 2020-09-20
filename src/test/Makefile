# Uncomment for debugging
DEBUG := 1
# Pretty build
# Q ?= @

CXX := g++
python := python3
PYTHON_HEADER_DIR := $(shell python -c 'from distutils.sysconfig import get_python_inc; print(get_python_inc())')
PYTORCH_INCLUDES := $(shell python -c 'from torch.utils.cpp_extension import include_paths; [print(p) for p in include_paths()]')
PYTORCH_LIBRARIES := $(shell python -c 'from torch.utils.cpp_extension import library_paths; [print(p) for p in library_paths()]')

CUDA_DIR := $(shell python -c 'from torch.utils.cpp_extension import _find_cuda_home; print(_find_cuda_home())')
WITH_ABI := $(shell python -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')
INCLUDE_DIRS := ./ $(CUDA_DIR)/include
INCLUDE_DIRS += $(PYTHON_HEADER_DIR)
INCLUDE_DIRS += $(PYTORCH_INCLUDES)

# Custom (MKL/ATLAS/OpenBLAS) include and lib directories.
# BLAS_INCLUDE := /path/to/your/blas
# BLAS_LIB := /path/to/your/blas

SRC_DIR := ./
OBJ_DIR := ./obj
CPP_SRCS := $(wildcard $(SRC_DIR)/*.cpp)
CU_SRCS := $(wildcard $(SRC_DIR)/*.cu)
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(CPP_SRCS))
CU_OBJS := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/cuda/%.o,$(CU_SRCS))
STATIC_LIB := $(OBJ_DIR)/libquant_impl.a


CUDA_ARCH := -gencode arch=compute_50,code=sm_50 \
		-gencode arch=compute_52,code=sm_52 \
		-gencode arch=compute_60,code=sm_60 \
		-gencode arch=compute_61,code=sm_61 \
		-gencode arch=compute_70,code=sm_70 \
		-gencode arch=compute_75,code=sm_75 \
		-gencode arch=compute_75,code=compute_75


LIBRARIES += stdc++ cudart c10 caffe2 torch torch_python caffe2_gpu


ifeq ($(DEBUG), 1)
	COMMON_FLAGS += -DDEBUG -g -O0
	NVCCFLAGS += -g -G # -rdc true
else
	COMMON_FLAGS += -DNDEBUG -O3
endif

WARNINGS := -Wall -Wno-sign-compare -Wcomment
INCLUDE_DIRS += $(BLAS_INCLUDE)
CXXFLAGS += -MMD -MP
COMMON_FLAGS += $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir)) \
	     -DTORCH_API_INCLUDE_EXTENSION_H -D_GLIBCXX_USE_CXX11_ABI=$(WITH_ABI)
CXXFLAGS += -pthread -fPIC -fwrapv -std=c++14 $(COMMON_FLAGS) $(WARNINGS)
NVCCFLAGS += -std=c++14 -ccbin=$(CXX) -Xcompiler -fPIC -use_fast_math $(COMMON_FLAGS)

default: $(STATIC_LIB)

$(OBJ_DIR):
	@ mkdir -p $@
	@ mkdir -p $@/cuda

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	@ echo CXX $<
	$(Q)$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJ_DIR)/cuda/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	@ echo NVCC $<
	$(Q)nvcc $(NVCCFLAGS) $(CUDA_ARCH) -M $< -o ${@:.o=.d} \
		-odir $(@D)
	$(Q)nvcc $(NVCCFLAGS) $(CUDA_ARCH) -c $< -o $@

$(STATIC_LIB): $(OBJS) $(CU_OBJS) | $(OBJ_DIR)
	$(RM) -f $(STATIC_LIB)
	$(RM) -rf build dist
	@ echo LD -o $@
	ar rc $(STATIC_LIB) $(OBJS) $(CU_OBJS)

build:
	$(python) setup.py build

upload:
	$(python) setup.py sdist bdist_wheel
	#twine upload dist/*

clean:
	$(RM) -rf build dist nnieqat.egg-info obj

test:
	nosetests -s tests/test_quant_impl.py --nologcapture

lint:
	pylint nnieqat --reports=n

lintfull:
	pylint nnieqat

install:
	$(python) setup.py install 

uninstall:
	$(python) setup.py install --record install.log
	cat install.log | xargs rm -rf 
	$(RM) install.log
