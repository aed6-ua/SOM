
__global__ void sumaVectores(float *vectorGPU  , float *vectorGPUResultado) {    
    int i =  blockDim.x * blockIdx.x + threadIdx.x;
    vectorGPUResultado[i]= vectorGPU[i] + i
}    




int main() { 
 
    int N = 5000    
    size_t size = N * sizeof(float);    
    
 
    float *vectorCPU = (float*)malloc(size);    
    float *vectorCPUResultado = (float*)malloc(size);   
    
    for (i = 0 ; i< N ; i++) {

        vectorCPU[i]=i;

    }  
    
    float *vectorGPU;    
    float *vectorGPUResultado;  
    cudaMalloc(&vectorGPU, size);    
    cudaMalloc(&vectorGPUResultado, size);  
   
    cudaMemcpy(vectorCPU, vectorGPU, size, cudaMemcpyHostToDevice);    

    
  
    int hilosPorBloque = 1;    
    int bloquesTotales = 5000;    
    sumaVectores<<<bloquesTotales, hilosPorBloque>>>(vectorGPU,vectorGPUResultado);    
    
  
    cudaMemcpy(vectorCPUResultado, vectorGPUResultado, size, cudaMemcpyDeviceToHost);    
    
   
    cudaFree(vectorGPU);    
    cudaFree(vectorGPUResultado);
    