#include <iostream>

#include "sherpa-onnx/c-api/c-api.h"

int main()
{
    std::cout
        << "sherpa-onnx version: "
        << SherpaOnnxGetVersionStr()
        << std::endl;

    return 0;
}