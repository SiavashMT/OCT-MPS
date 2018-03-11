#!/bin/bash

cat << EOF > /tmp/detect_compute_capability.cu
#include <stdio.h>
int main()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop,0);
    int v = prop.major * 10 + prop.minor;
    printf("-gencode arch=compute_%d,code=sm_%d",v,v);
}
EOF

# Compile
/usr/local/cuda/bin/nvcc /tmp/detect_compute_capability.cu -o /tmp/detect_compute_capability

# Probe the card
/tmp/detect_compute_capability

# Cleanup
rm /tmp/detect_compute_capability.cu
rm /tmp/detect_compute_capability