#include "fake_quantize.h"
__global__ void fake_quantize_kernel_cuda(float* __restrict__ a,
                                            float* o, int size,
                                            float* max_entry,
                                            int bit_width) {
    if(bit_width!=8) bit_width =16;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index < size) {
        if((*max_entry) < 1e-15 && (*max_entry) > -1e-15){
            o[index] = 0;
            return;
        }

        if(bit_width == 8){
            float data_max = (*max_entry);
            int max_entry_qdata_int =  floorf(__log2f(data_max) * 16) + 1;
            data_max = __powf(2, __fdividef(max_entry_qdata_int, 16));
            float data_max_floor = __powf(2, __fdividef(max_entry_qdata_int-1, 16));

            if(a[index] <= data_max_floor * 0.0020395972313035  // exp(ln(256) / 128) / 512= 2^(1/16-9) = 1.0442737824274 /512 = 0.0020395972313035
                && a[index] > - data_max * 0.0020395972313035){  
                o[index] = 0;
                return;
            }

            //int qdata_int = (int)(log(256 * a[index] / data_max ) / 0.04332169878499658);  //ln(256) / 128 =  0.04332169878499658
            int qdata_int = 0;
            if(a[index] > 0){
                qdata_int = rintf(__fdividef(  __logf(__fdividef(256* a[index],data_max)), 0.04332169878499658));  //ln(256) / 128 =  0.04332169878
                if(qdata_int > 127) qdata_int = 127;
                else if(qdata_int < 0) qdata_int = 0;   
                o[index] =  __fdividef(data_max , 256.0) *  __expf(qdata_int*0.04332169878499658);   
            }
            else{
                qdata_int = - rintf(__fdividef(  __logf(__fdividef(- 256* a[index], data_max)), 0.04332169878499658));  //ln(256) / 128 =  0.04332169878
                if(qdata_int < -127) qdata_int = -127;
                else if(qdata_int >-1) qdata_int = -1;
                o[index] = - __fdividef(data_max , 256.0) * __expf(- qdata_int*0.04332169878499658);
            }

        }
        else{
            float data_max = (*max_entry);
            int max_entry_qdata_int =  floorf(__log2f(data_max) * 128) + 1;
            data_max = __powf(2, __fdividef(max_entry_qdata_int, 128));
            float data_max_floor = __powf(2, __fdividef(max_entry_qdata_int-1, 16));

            
            if(a[index] < data_max_floor *0.0019537861485404  //exp(ln(2^16)/(2^15)) / 512 = 0.0019537861485404
                && a[index] > - data_max * 0.0019537861485404){ 
                o[index] = 0;
                return;
            }

            int qdata_int = 0;
            if(a[index] > 0){
                qdata_int = rintf(__fdividef(  __logf(__fdividef(65536* a[index], data_max)), 0.00033845077175779)); 
                if(qdata_int > 32767) qdata_int = 32767;
                else if(qdata_int <0) qdata_int = 0;
                o[index] =  __fdividef(data_max , 65536.0) * __expf(qdata_int * 0.00033845077175779); 
            }
            else{
                qdata_int = - rintf(__fdividef(  __logf(__fdividef(- 65536* a[index], data_max)), 0.00033845077175779));
                if(qdata_int < -32767) qdata_int = -32767;
                else if(qdata_int >-1) qdata_int = -1;
                o[index] = - __fdividef(data_max , 65536.0) * __expf(- qdata_int * 0.00033845077175779);  
            }
        }

    }
}


Tensor fake_quantize_cuda(Tensor a, int bit_width) {
    auto o = at::zeros_like(a);
    int64_t size = a.numel();
  
    Tensor max_entry = at::max(at::abs(a));
    int blockSize = 1024;
    int blockNums = (size + blockSize - 1) / blockSize;
  
    fake_quantize_kernel_cuda<<<blockNums, blockSize>>>(a.data_ptr<float>(),
                                                        o.data_ptr<float>(),
                                                        size,
                                                        max_entry.data_ptr<float>(),
                                                        bit_width);
    return o;
  }

