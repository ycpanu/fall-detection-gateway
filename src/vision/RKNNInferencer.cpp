#include <fstream>
#include <cstring>
#include <cmath>
#include <algorithm>

#include "fall-detection/vision/RKNNInferencer.hpp"
#include "fall-detection/utils/SysLogger.hpp"

namespace fall_detection
{
    namespace vision
    {
        RKNNInferencer::RKNNInferencer(const std::string& modelPath, float confThreshold, float nmsThreshold)
            : modelPath_(modelPath), confThreshold_(confThreshold), nmsThreshold_(nmsThreshold),
              ctx_(0), isInitialized_(false), inputAttrs_(nullptr), outputAttrs_(nullptr),
              numInput_(0), numOutput_(0) {}

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
                LOG_INFO("输出张量 {}: n_dims={}, dims=[{}, {}, {}, {}], n_elems={}", i,
                        outputAttrs_[i].n_dims, outputAttrs_[i].dims[0], outputAttrs_[i].dims[1],
                        outputAttrs_[i].dims[2], outputAttrs_[i].dims[3], outputAttrs_[i].n_elems);
            }
            LOG_INFO("成功加载 RKNN 模型！模型期望输入尺寸：宽={} 高={} 通道={}",inputAttrs_[0].dims[1], inputAttrs_[0].dims[2], inputAttrs_[0].dims[3]);

            isInitialized_ = true;
            return true;
        }

        bool RKNNInferencer::detect(const cv::Mat& frame, std::vector<DetectResult>& results)
        {
            if (!isInitialized_)
            {
                LOG_ERROR("调用 detect 前必须先成功执行 init()！");
                return false;
            }
            if (frame.empty())
            {
                LOG_WARN("detect 收到空帧，跳过本次推理");
                return false;
            }

            // 1. 图像预处理，获取模型需要的宽高
            int reqWidth = inputAttrs_[0].dims[1];
            int reqHeight = inputAttrs_[0].dims[2];

            cv::Mat rgbFrame;
            // OpenCV 默认 BGR，需要转换成 RGB
            cv::cvtColor(frame, rgbFrame, cv::COLOR_BGR2RGB);

            // Letterbox 缩放：保持宽高比，不足部分用灰边(114)填充，避免画面拉伸变形
            float scale = std::min(static_cast<float>(reqWidth) / rgbFrame.cols,
                                static_cast<float>(reqHeight) / rgbFrame.rows);
            int newW = static_cast<int>(std::round(rgbFrame.cols * scale));
            int newH = static_cast<int>(std::round(rgbFrame.rows * scale));
            int padW = (reqWidth - newW) / 2;
            int padH = (reqHeight - newH) / 2;

            cv::Mat resized;
            cv::resize(rgbFrame, resized, cv::Size(newW, newH));
            cv::Mat letterboxed(reqHeight, reqWidth, CV_8UC3, cv::Scalar(114, 114, 114));
            resized.copyTo(letterboxed(cv::Rect(padW, padH, newW, newH)));

            // 2. NPU 硬件推理
            rknn_input inputs[1];
            memset(inputs, 0, sizeof(inputs));
            inputs[0].index = 0;
            inputs[0].type = RKNN_TENSOR_UINT8;     // 图片像素格式一般为 unit8
            inputs[0].size = letterboxed.cols * letterboxed.rows * letterboxed.channels();
            inputs[0].fmt = RKNN_TENSOR_NHWC;       // 数据排布格式 (N:批次, H:高, W:宽, C:通道)
            inputs[0].buf = letterboxed.data;       // 将OpenCV 的图像数据指针直接给 NPU
            inputs[0].pass_through = 0;

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

            float* outData = static_cast<float*>(outputs[0].buf);
            std::vector<DetectResult> candidates;

            // YOLOv8-Pose 每个锚框的属性布局（属性切片，跨度为 NUM_ANCHORS）：
            // [0..3] 边框 cx,cy,w,h   [4] 类别置信度(person)   [5..55] 17 个关键点(x,y,conf)
            const int KPT_OFFSET = 4 + NUM_CLASSES;                 // 关键点数据起始索引 = 5
            const int TOTAL_ATTR = KPT_OFFSET + NUM_KEYPOINTS * 3;  // 4 + 1 + 17*3 = 56
            const int NUM_ANCHORS = static_cast<int>(outputAttrs_[0].n_elems) / TOTAL_ATTR;

            for (int i = 0; i < NUM_ANCHORS; i++)
            {
                // Pose 模型只有 person 一类，类别置信度即第 4 个属性
                float clsConf = outData[4 * NUM_ANCHORS + i];
                if (clsConf <= confThreshold_)
                {
                    continue;
                }

                // 解析边框中心点与宽高
                float cx = outData[0 * NUM_ANCHORS + i];
                float cy = outData[1 * NUM_ANCHORS + i];
                float w  = outData[2 * NUM_ANCHORS + i];
                float h  = outData[3 * NUM_ANCHORS + i];

                // 边框解码：模型输出是相对 reqWidth x reqHeight 的像素，
                // 映射回原图需先减 letterbox 灰边偏移，再除以缩放比例
                DetectResult box;
                box.classId = 0;   // person
                box.confidence = clsConf;
                box.x = static_cast<int>((cx - w / 2.0f - padW) / scale);
                box.y = static_cast<int>((cy - h / 2.0f - padH) / scale);
                box.width = static_cast<int>(w / scale);
                box.height = static_cast<int>(h / scale);

                // 解码 17 个骨骼关键点（关键点坐标同样需逆映射到原图）
                box.keypoints.reserve(NUM_KEYPOINTS);
                for (int k = 0; k < NUM_KEYPOINTS; k++)
                {
                    KeyPoint kp;
                    kp.x = static_cast<int>((outData[(KPT_OFFSET + k * 3 + 0) * NUM_ANCHORS + i] - padW) / scale);
                    kp.y = static_cast<int>((outData[(KPT_OFFSET + k * 3 + 1) * NUM_ANCHORS + i] - padH) / scale);
                    kp.confidence = outData[(KPT_OFFSET + k * 3 + 2) * NUM_ANCHORS + i];
                    box.keypoints.push_back(kp);
                }
   
                candidates.push_back(box);
            }

            // 非极大值抑制(NMS) - 消除重影
            nms(candidates, results);

            // 诊断日志：定位“检测不到摔倒”时问题在后处理还是判断引擎
            if (results.empty())
            {
                LOG_TRACE("本帧未解析出有效人体（候选框 {} 个，confThreshold={}）", candidates.size(), confThreshold_);
            }
            else
            {
                LOG_TRACE("本帧解析出 {} 个人体，第一个含 {} 个关键点",
                          results.size(), results[0].keypoints.size());
            }

            // 释放 NPU 输出内存
            rknn_outputs_release(ctx_, numOutput_, outputs);

            return true;
        }

        void RKNNInferencer::nms(std::vector<DetectResult>& inputBoxes, std::vector<DetectResult>& outputBoxes)
        {
            outputBoxes.clear();

            // 按照置信度从高到低排序，优先保留最确定的预测框
            std::sort(inputBoxes.begin(), inputBoxes.end(), [](const DetectResult& a, const DetectResult& b)
            {
                return a.confidence > b.confidence;
            });

            std::vector<bool> isSuppressed(inputBoxes.size(), false);

            for (size_t i = 0; i < inputBoxes.size(); i++)
            {
                if (isSuppressed[i]) continue;

                outputBoxes.push_back(inputBoxes[i]);   // 保留最高分的框

                // 检查后面所有的框，如果和当前框的 IoU 超过阈值，直接抹杀
                for (size_t j = i + 1; j < inputBoxes.size(); j++)
                {
                    if (!isSuppressed[j] && inputBoxes[i].classId == inputBoxes[j].classId)
                    {
                        float iou = calculateIoU(inputBoxes[i], inputBoxes[j]);
                        if (iou > nmsThreshold_)
                        {
                            isSuppressed[j] = true;
                        }
                    }
                }
            }
        }

        float RKNNInferencer::calculateIoU(const DetectResult& box1, const DetectResult& box2)
        {
            // 计算交集的坐标
            int x1 = std::max(box1.x, box2.x);
            int y1 = std::max(box1.y, box2.y);
            int x2 = std::min(box1.x + box1.width, box2.x + box2.width);
            int y2 = std::min(box1.y + box1.height, box2.y + box2.height);

            // 如果没有交集
            if (x1 >= x2 || y1 >= y2) return 0.0f;

            float intersectionArea = static_cast<float>((x2 - x1) * (y2 - y1));
            float box1Area = static_cast<float>(box1.width * box1.height);
            float box2Area = static_cast<float>(box2.width * box2.height);

            // 并集面积为 0（两个框都没有有效面积）时直接返回 0，避免除零
            float unionArea = box1Area + box2Area - intersectionArea;
            if (unionArea <= 0.0f) return 0.0f;

            // 经典的 IoU 公式：交集面积 / 并集面积
            return intersectionArea / unionArea;
        }
    }
}