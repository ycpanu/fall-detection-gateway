#include <fstream>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>

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

            file.seekg(0, std::ios::end);
            *modelSize = file.tellg();
            file.seekg(0, std::ios::beg);

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

            // 1. 初始化 RKNN 环境
            int ret = rknn_init(&ctx_, modelData, modelSize, 0, NULL);
            delete[] modelData; 

            if (ret < 0)
            {
                LOG_ERROR("rknn_init 失败！错误码：{}", ret);
                return false;
            }

            // 2. 严格校验查询返回值
            rknn_input_output_num ioNum;
            ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &ioNum, sizeof(ioNum));
            if (ret < 0)
            {
                LOG_ERROR("查询输入输出张量数量失败！错误码：{}", ret);
                return false;
            }
            
            numInput_ = ioNum.n_input;
            numOutput_ = ioNum.n_output;

            inputAttrs_ = new rknn_tensor_attr[numInput_];
            outputAttrs_ = new rknn_tensor_attr[numOutput_];

            // 3. 获取输入张量属性
            for (int i = 0; i < numInput_; i++)
            {
                memset(&inputAttrs_[i], 0, sizeof(rknn_tensor_attr));
                inputAttrs_[i].index = i;
                ret = rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &(inputAttrs_[i]), sizeof(rknn_tensor_attr));
                if (ret < 0)
                {
                    LOG_ERROR("查询输入张量 {} 属性失败！错误码：{}", i, ret);
                    return false;
                }
            }

            // 4. 获取输出张量属性
            for (int i = 0; i < numOutput_; i++)
            {
                memset(&outputAttrs_[i], 0, sizeof(rknn_tensor_attr));
                outputAttrs_[i].index = i;
                ret = rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &(outputAttrs_[i]), sizeof(rknn_tensor_attr));
                if (ret < 0)
                {
                    LOG_ERROR("查询输出张量 {} 属性失败！错误码：{}", i, ret);
                    return false;
                }
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

            int reqWidth = inputAttrs_[0].dims[1];
            int reqHeight = inputAttrs_[0].dims[2];

            cv::Mat rgbFrame;
            cv::cvtColor(frame, rgbFrame, cv::COLOR_BGR2RGB);

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

            rknn_input inputs[1];
            memset(inputs, 0, sizeof(inputs));
            inputs[0].index = 0;
            inputs[0].type = RKNN_TENSOR_UINT8;     
            inputs[0].size = letterboxed.cols * letterboxed.rows * letterboxed.channels();
            inputs[0].fmt = RKNN_TENSOR_NHWC;       
            inputs[0].buf = letterboxed.data;       
            inputs[0].pass_through = 0;

            int ret = rknn_inputs_set(ctx_, numInput_, inputs);
            if (ret < 0)
            {
                LOG_ERROR("rknn_inputs_set 失败！错误码：{}", ret);
                return false;
            }

            ret = rknn_run(ctx_, NULL);
            if (ret < 0)
            {
                LOG_ERROR("rknn_run 推理失败！错误码：{}", ret);
                return false;
            }

            // 修复 VLA 变长数组问题，使用标准的 std::vector 容器
            std::vector<rknn_output> outputs(numOutput_);
            memset(outputs.data(), 0, sizeof(rknn_output) * numOutput_);
            for (int i = 0; i < numOutput_; i++)
            {
                outputs[i].want_float = 1;      
            }

            // 传入 .data() 获取底层指针
            ret = rknn_outputs_get(ctx_, numOutput_, outputs.data(), NULL);
            if (ret < 0)
            {
                LOG_ERROR("rknn_outputs_get 失败！错误码：{}", ret);
                return false;
            }

            float* outData = static_cast<float*>(outputs[0].buf);
            std::vector<DetectResult> candidates;

            const int KPT_OFFSET = 4 + NUM_CLASSES;                 
            const int TOTAL_ATTR = KPT_OFFSET + NUM_KEYPOINTS * 3;  
            const int NUM_ANCHORS = static_cast<int>(outputAttrs_[0].n_elems) / TOTAL_ATTR;

            for (int i = 0; i < NUM_ANCHORS; i++)
            {
                float clsConf = outData[4 * NUM_ANCHORS + i];
                if (clsConf <= confThreshold_) continue;

                float cx = outData[0 * NUM_ANCHORS + i];
                float cy = outData[1 * NUM_ANCHORS + i];
                float w  = outData[2 * NUM_ANCHORS + i];
                float h  = outData[3 * NUM_ANCHORS + i];

                DetectResult box;
                box.classId = 0;   
                box.confidence = clsConf;
                box.x = static_cast<int>((cx - w / 2.0f - padW) / scale);
                box.y = static_cast<int>((cy - h / 2.0f - padH) / scale);
                box.width = static_cast<int>(w / scale);
                box.height = static_cast<int>(h / scale);

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

            nms(candidates, results);

            if (results.empty())
            {
                LOG_TRACE("本帧未解析出有效人体（候选框 {} 个，confThreshold={}）", candidates.size(), confThreshold_);
            }
            else
            {
                LOG_TRACE("本帧解析出 {} 个人体，第一个含 {} 个关键点",
                          results.size(), results[0].keypoints.size());
            }

            // 传入 .data() 获取底层指针
            rknn_outputs_release(ctx_, numOutput_, outputs.data());

            return true;
        }

        void RKNNInferencer::nms(std::vector<DetectResult>& inputBoxes, std::vector<DetectResult>& outputBoxes)
        {
            outputBoxes.clear();

            std::sort(inputBoxes.begin(), inputBoxes.end(), [](const DetectResult& a, const DetectResult& b)
            {
                return a.confidence > b.confidence;
            });

            std::vector<bool> isSuppressed(inputBoxes.size(), false);

            for (size_t i = 0; i < inputBoxes.size(); i++)
            {
                if (isSuppressed[i]) continue;

                outputBoxes.push_back(inputBoxes[i]);   

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
            int x1 = std::max(box1.x, box2.x);
            int y1 = std::max(box1.y, box2.y);
            int x2 = std::min(box1.x + box1.width, box2.x + box2.width);
            int y2 = std::min(box1.y + box1.height, box2.y + box2.height);

            if (x1 >= x2 || y1 >= y2) return 0.0f;

            float intersectionArea = static_cast<float>((x2 - x1) * (y2 - y1));
            float box1Area = static_cast<float>(box1.width * box1.height);
            float box2Area = static_cast<float>(box2.width * box2.height);

            float unionArea = box1Area + box2Area - intersectionArea;
            if (unionArea <= 0.0f) return 0.0f;

            return intersectionArea / unionArea;
        }
    }
}