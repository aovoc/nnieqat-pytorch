#include "fake_quantize.h"

__global__ void fake_quantize_kernel_cuda(float* __restrict__ a,
                                            float* o, int size,
                                            float* max_entry,
                                            int bit_width) {
    if(bit_width!=8) bit_width =16;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        if((*max_entry) < 1e-20 && (*max_entry) > -1e-20 || 
            a[index] < 1e-20 && a[index] > -1e-20){
            o[index] = a[index];
            return;
        }

        if(bit_width == 8){
            int max_entry_qdata_int =  floorf(__log2f((*max_entry)) * 16);
            (*max_entry) = __powf(2, __fdividef(max_entry_qdata_int, 16));

            //int qdata_int = (int)(log(256 * a[index] / (*max_entry) ) / 0.04332169878499658);  //ln(256) / 128 =  0.04332169878499658
            int qdata_int = 0;
            if(a[index] > 0)
                qdata_int = rintf(__fdividef(  __logf(__fdividef(256* a[index], (*max_entry))), 0.04332169878499658));  //ln(256) / 128 =  0.04332169878
            else
                qdata_int = - rintf(__fdividef(  __logf(__fdividef(- 256* a[index], (*max_entry))), 0.04332169878499658));  //ln(256) / 128 =  0.04332169878

            if(qdata_int > 128) qdata_int = 128;
            else if(qdata_int < -128) qdata_int = -128;

            //o[index] =  (*max_entry) / 256.0 * exp(qdata_int * 0.04332169878499658); 
            if(qdata_int > 0){
                o[index] =  __fdividef((*max_entry) , 256.0) * __expf(qdata_int * 0.04332169878499658);   
            }else if(qdata_int == 0){
                o[index] = 0;
            }else{
                o[index] = - __fdividef((*max_entry) , 256.0) * __expf(- qdata_int * 0.04332169878499658);
            }

        }
        else{

            int max_entry_qdata_int =  floorf(__log2f((*max_entry)) * 128);
            (*max_entry) = __powf(2, __fdividef(max_entry_qdata_int, 128));

            int qdata_int = 0;
            if(a[index] > 0)
                qdata_int = rintf(__fdividef(  __logf(__fdividef(65536* a[index], (*max_entry))), 0.00033845077175779));  //ln(2^16)/(2^15) = 0.00033845077175779
            else
                qdata_int = - rintf(__fdividef(  __logf(__fdividef(- 65536* a[index], (*max_entry))), 0.00033845077175779));  //ln(2^16)/(2^15) = 0.00033845077175779   
            
            if(qdata_int > 32768) qdata_int = 32768;
            else if(qdata_int < -32768) qdata_int = -32768;        
            
            if(qdata_int > 0){
                o[index] =  __fdividef((*max_entry) , 65536.0) * __expf(qdata_int * 0.00033845077175779);  
            }else if(qdata_int == 0){
                o[index] = 0;
            }else{
                o[index] = - __fdividef((*max_entry) , 65536.0) * __expf(- qdata_int * 0.00033845077175779);  
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

