#include <fstream>
#include <cstring>

#include "fall-detection/vision/RKNNInferencer.hpp"
#include "fall-detection/utils/SysLogger.hpp"

namespace fall_detection
{
    RKNNInferencer::RKNNInferencer(const std::string& modelPath) : modelPath_(modelPath), ctx_(0), isInitialized_(false), inputAttrs_(nullptr), outputAttrs_(nullptr), numInput_(0), numOutput_(0) {}

    RKNNInferencer::~RKNNInferencer()
    {
        // 析构必须销毁 RKNN 上下文，释放开发板的 NPU 内存
        if (ctx_ > 0)
        {
            rknn_destroy(ctx_);
            LOG_INFO("已安全释放 RKNN NPU 上下文与硬件资源。");
        }

        if (inputAttrs_ != nullptr)
        {
            delete[] inputAttrs_;
        }

        if (outputAttrs_ != nullptr)
        {
            delete[] outputAttrs_;
        }
    }

    unsigned char* RKNNInferencer::loadModelFile(const char* filename, int* modelSize)
    {
        std::ifstream file(filename, std::ios::in | std::ios::binary);
        if (!file.is_open())
        {
            LOG_ERROR("无法打开模型文件: {}", filename);
            return nullptr;
        }

        // 获取文件大小
        file.seekg(0, std::ios::end);
        *modelSize = file.tellg();
        file.seekg(0, std::ios::beg);

        // 分配内存并读取数据
        unsigned char* modelData = new unsigned char[*modelSize];
        file.read(reinterpret_cast<char*>(modelData), *modelSize);
        file.close();

        return modelData;
    }

    bool RKNNInferencer::init()
    {
        if (isInitialized_)
        {
            LOG_INFO("RKNN 模型已初始化，无需重复操作。");
            return true;
        }

        int modelSize = 0;
        unsigned char* modelData = loadModelFile(modelPath_.c_str(), &modelSize);

        if (modelData == nullptr)
        {
            LOG_ERROR("加载 .rknn 模型数据失败！");
            return false;
        }

        // 1. 初始化 RKNN 环境，分配 NPU 资源
        int ret = rknn_init(&ctx_, modelData, modelSize, 0, NULL);
        delete[] modelData; // 数据已载入 NPU，可释放本地内存

        if (ret < 0)
        {
            LOG_ERROR("rknn_init 失败！错误码：{}", ret);
            return false;
        }

        // 2. 查询模型的输入/输出张量(Tensor)数量
        rknn_input_output_num ioNum;
        rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &ioNum, sizeof(ioNum));
        numInput_ = ioNum.n_input;
        numOutput_ = ioNum.n_output;

        inputAttrs_ = new rknn_tensor_attr[numInput_];
        outputAttrs_ = new rknn_tensor_attr[numOutput_];

        // 3. 获取输入张量属性，告诉我们需要输入多大的图片
        for (int i = 0; i < numInput_; i++)
        {
            memset(&inputAttrs_[i], 0, sizeof(rknn_tensor_attr));
            inputAttrs_[i].index = i;
            rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &(inputAttrs_[i]), sizeof(rknn_tensor_attr));

        }

        // 4. 获取输出张量的属性，告诉我们会输出怎样格式的结果
        for (int i = 0; i < numOutput_; i++)
        {
            memset(&outputAttrs_[i], 0, sizeof(rknn_tensor_attr));
            outputAttrs_[i].index = i;
            rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &(outputAttrs_[i]), sizeof(rknn_tensor_attr));

            LOG_INFO("成功加载 RKNN 模型！模型期望输入尺寸：宽={} 高={} 通道={}",inputAttrs_[0].dims[1], inputAttrs_[0].dims[2], inputAttrs_[0].dims[3]);

            isInitialized_ = true;
            return true;
        }
    }
    bool RKNNInferencer::detect(const cv::Mat& frame, std::vector<DetectResult>& results)
    {
        if (!isInitialized_)
        {
            LOG_ERROR("调用 detect 前必须先成功执行 init()！");
            return false;
        }

        // 1. 图像预处理，获取模型需要的宽高
        int reqWidth = inputAttrs_[0].dims[1];
        int reqHeight = inputAttrs_[0].dims[2];

        cv::Mat rgbFrame, resizedFrame;
        // OpenCV 默认 BGR，需要转换成 RGB
        cv::cvtColor(frame, rgbFrame, cv::COLOR_BGR2RGB);
        // 将原始图像强制缩放到模型所需要的尺寸
        cv::resize(rgbFrame, resizedFrame, cv::Size(reqWidth, reqHeight));

        // 2. NPU 硬件推理
        rknn_input inputs[1];
        memset(inputs, 0, sizeof(inputs));
        inputs[0].index = 0;
        inputs[0].type = RKNN_TENSOR_UINT8;     // 图片像素格式一般为 unit8
        inputs[0].size = resizedFrame.cols * resizedFrame.rows * resizedFrame.channels();
        inputs[0].fmt = RKNN_TENSOR_NHWC;       // 数据排布格式 (N:批次, H:高, W:宽, C:通道)
        inputs[0].buf = resizedFrame.data;      // 将OpenCV 的图像数据指针直接给 NPU

        // 将数据推入 NPU 显存
        int ret = rknn_inputs_set(ctx_, numInput_, inputs);
        if (ret < 0)
        {
            LOG_ERROR("rknn_inputs_set 失败！错误码：{}", ret);
            return false;
        }

        // 一键启动极速硬件计算
        ret = rknn_run(ctx_, NULL);
        if (ret < 0)
        {
            LOG_ERROR("rknn_run 推理失败！错误码：{}", ret);
            return false;
        }

        // 3. 获取解析结果
        rknn_output outputs[numOutput_];
        memset(outputs, 0, sizeof(outputs));
        for (int i = 0; i < numOutput_; i++)
        {
            outputs[i].want_float = 1;      // 强制要求 NPU 把 INT8 的结果反量化为 float32
        }

        // 从 NPU 取回计算结果
        ret = rknn_outputs_get(ctx_, numOutput_, outputs, NULL);
        if (ret < 0)
        {
            LOG_ERROR("rknn_outputs_get 失败！错误码：{}", ret);
            return false;
        }

        // 在实际的 YOLOv8 部署中，这里需要数百行代码来执行“非极大值抑制(NMS)”和“锚框解码”。
        // 为了保持底层基建框架的纯粹性，这部分复杂的数学解析我们会在后续的 
        // 规则引擎 (FallRuleEngine) 中分离处理。

        // 解析完毕后，必须释放输出内存，防止内存泄露
        rknn_outputs_release(ctx_, numOutput_, outputs);

        return true;
    }
}