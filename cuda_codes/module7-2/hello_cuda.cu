#include <stdio.h>

// empty function kernel() qualified with __global__
__global__ void kernel(void)
{
}

int main(void)
{
    // A call to the empty function, with additional <<<1,1>>>
    kernel<<<1,1>>>();

    printf("Hello, World! \n");
    return 0;
}
