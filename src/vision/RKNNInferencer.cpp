#include <fstream>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>

#include "fall-detection/vision/RKNNInferencer.hpp"
#include "fall-detection/utils/SysLogger.hpp"

#include <chrono>
#include <limits>

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

            LOG_INFO("正在加载 RKNN 模型: {}", modelPath_);
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

            rknn_sdk_version sdkVersion{};
            ret = rknn_query(ctx_, RKNN_QUERY_SDK_VERSION, &sdkVersion, sizeof(sdkVersion));
            if (ret == 0)
            {
                LOG_INFO("RKNN Runtime={}, NPU Driver={}", sdkVersion.api_version, sdkVersion.drv_version);
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
                LOG_INFO("输入张量 {}: n_dims={}, dims=[{}, {}, {}, {}], fmt={}, type={}, qnt_type={}, scale={}, zp={}",
                         i, inputAttrs_[i].n_dims, inputAttrs_[i].dims[0], inputAttrs_[i].dims[1],
                         inputAttrs_[i].dims[2], inputAttrs_[i].dims[3],
                         get_format_string(inputAttrs_[i].fmt), get_type_string(inputAttrs_[i].type),
                         get_qnt_type_string(inputAttrs_[i].qnt_type), inputAttrs_[i].scale, inputAttrs_[i].zp);
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
                LOG_INFO("输出张量 {}: n_dims={}, dims=[{}, {}, {}, {}], n_elems={}, fmt={}, type={}, qnt_type={}, scale={}, zp={}", i,
                        outputAttrs_[i].n_dims, outputAttrs_[i].dims[0], outputAttrs_[i].dims[1],
                        outputAttrs_[i].dims[2], outputAttrs_[i].dims[3], outputAttrs_[i].n_elems,
                        get_format_string(outputAttrs_[i].fmt), get_type_string(outputAttrs_[i].type),
                        get_qnt_type_string(outputAttrs_[i].qnt_type), outputAttrs_[i].scale, outputAttrs_[i].zp);
            }
            
            if (numInput_ != 1 || inputAttrs_[0].n_dims != 4 ||
                (inputAttrs_[0].fmt != RKNN_TENSOR_NHWC && inputAttrs_[0].fmt != RKNN_TENSOR_NCHW))
            {
                LOG_ERROR("需要单个四维 NHWC/NCHW 图像输入");
                return false;
            }
            const bool nhwc = inputAttrs_[0].fmt == RKNN_TENSOR_NHWC;
            inputSize_ = cv::Size(inputAttrs_[0].dims[nhwc ? 2 : 3], inputAttrs_[0].dims[nhwc ? 1 : 2]);
            const int channels = inputAttrs_[0].dims[nhwc ? 3 : 1];
            if (inputAttrs_[0].dims[0] != 1 || channels != 3 || inputSize_.width <= 0 || inputSize_.height <= 0)
            {
                LOG_ERROR("需要 batch=1、三通道且尺寸有效的图像输入");
                return false;
            }
            LOG_INFO("成功加载 RKNN 模型！模型期望输入尺寸：宽={} 高={} 通道={}", inputSize_.width, inputSize_.height, channels);

            isInitialized_ = true;
            return true;
        }

        bool RKNNInferencer::detect(const cv::Mat& frame, std::vector<DetectResult>& results)
        {
            constexpr int WARMUP_FRAMES = 20;
            constexpr int REPORT_FRAMES = 100;

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

            const auto preprocessStart = std::chrono::steady_clock::now();
            const int reqWidth = inputSize_.width;
            const int reqHeight = inputSize_.height;

            float scale = std::min(static_cast<float>(reqWidth) / frame.cols,
                                static_cast<float>(reqHeight) / frame.rows);
            int newW = static_cast<int>(std::round(frame.cols * scale));
            int newH = static_cast<int>(std::round(frame.rows * scale));
            int padW = (reqWidth - newW) / 2;
            int padH = (reqHeight - newH) / 2;

            // 消除魔法数字，定义为明确的常量
            constexpr int LETTERBOX_PAD_COLOR = 114;

            // 先缩放再交换通道，避免对原始大图转色；复用缓冲区并直接写入目标 ROI。
            cv::resize(frame, resizedBgr_, cv::Size(newW, newH));
            letterboxedRgb_.create(reqHeight, reqWidth, CV_8UC3);
            letterboxedRgb_.setTo(cv::Scalar::all(LETTERBOX_PAD_COLOR));
            cv::Mat imageRegion = letterboxedRgb_(cv::Rect(padW, padH, newW, newH));
            cv::cvtColor(resizedBgr_, imageRegion, cv::COLOR_BGR2RGB);

            rknn_input inputs[1];
            memset(inputs, 0, sizeof(inputs));
            inputs[0].index = 0;
            inputs[0].type = RKNN_TENSOR_UINT8;     
            inputs[0].size = letterboxedRgb_.total() * letterboxedRgb_.elemSize();
            inputs[0].fmt = RKNN_TENSOR_NHWC;       
            inputs[0].buf = letterboxedRgb_.data;
            inputs[0].pass_through = 0;

            const auto preprocessEnd = std::chrono::steady_clock::now();
            int ret = rknn_inputs_set(ctx_, numInput_, inputs);
            if (ret < 0)
            {
                LOG_ERROR("rknn_inputs_set 失败！错误码：{}", ret);
                return false;
            }

            auto rknnStart = std::chrono::steady_clock::now();

            ret = rknn_run(ctx_, NULL);

            auto rknnEnd = std::chrono::steady_clock::now();

            if (ret < 0)
            {
                LOG_ERROR("rknn_run 推理失败！错误码：{}", ret);
                return false;
            }

            double rknnMs =
                std::chrono::duration<double, std::milli>(rknnEnd - rknnStart).count();

            // 修复 VLA 变长数组问题，使用标准的 std::vector 容器
            std::vector<rknn_output> outputs(numOutput_);
            memset(outputs.data(), 0, sizeof(rknn_output) * numOutput_);
            for (int i = 0; i < numOutput_; i++)
            {
                outputs[i].index = i;
                // Runtime 将量化输出转换为 float，下面按 float* 解析，无需再次反量化。
                outputs[i].want_float = 1;      
            }

            // 传入 .data() 获取底层指针
            ret = rknn_outputs_get(ctx_, numOutput_, outputs.data(), NULL);
            const auto outputEnd = std::chrono::steady_clock::now();
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
            float maxPersonConfidence = -std::numeric_limits<float>::infinity();
            int nonFiniteScores = 0;

            for (int i = 0; i < NUM_ANCHORS; i++)
            {
                float clsConf = outData[4 * NUM_ANCHORS + i];
                if (std::isfinite(clsConf))
                {
                    maxPersonConfidence = std::max(maxPersonConfidence, clsConf);
                }
                else
                {
                    ++nonFiniteScores;
                    continue;
                }
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

            // INFO 级别只打印前 3 帧及每 100 帧，便于与 ONNX 的 person_scores.max() 对比。
            ++diagnosticFrameCount_;
            if (diagnosticFrameCount_ <= 3 || diagnosticFrameCount_ % 100 == 0)
            {
                LOG_INFO("[POSE诊断] frame={}, input={}x{}, want_float=1, output_bytes={}, max_person_conf={}, non_finite_scores={}, threshold={}, candidates={}, persons={}",
                         diagnosticFrameCount_, reqWidth, reqHeight, outputs[0].size,
                         maxPersonConfidence, nonFiniteScores, confThreshold_, candidates.size(), results.size());
            }

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

            const auto postprocessEnd = std::chrono::steady_clock::now();
            if (warmupCount_ < WARMUP_FRAMES)
            {
                ++warmupCount_;
            }
            else
            {
                const auto milliseconds = [](auto duration) {
                    return std::chrono::duration<double, std::milli>(duration).count();
                };
                const std::array<double, 5> stages = {
                    milliseconds(preprocessEnd - preprocessStart),
                    milliseconds(rknnStart - preprocessEnd),
                    rknnMs,
                    milliseconds(outputEnd - rknnEnd),
                    milliseconds(postprocessEnd - outputEnd)
                };
                for (size_t i = 0; i < stages.size(); ++i) stageTotalMs_[i] += stages[i];
                minRknnMs_ = std::min(minRknnMs_, rknnMs);
                maxRknnMs_ = std::max(maxRknnMs_, rknnMs);
                if (++measuredCount_ >= REPORT_FRAMES)
                {
                    LOG_INFO("[PERF] RKNN rknn_run: avg={:.2f} ms, min={:.2f} ms, max={:.2f} ms, samples={}",
                             stageTotalMs_[2] / measuredCount_, minRknnMs_, maxRknnMs_, measuredCount_);
                    LOG_INFO("[PERF] RKNN stages: preprocess={:.2f} ms, inputs_set={:.2f} ms, run={:.2f} ms, outputs_get={:.2f} ms, postprocess={:.2f} ms, samples={}",
                             stageTotalMs_[0] / measuredCount_, stageTotalMs_[1] / measuredCount_,
                             stageTotalMs_[2] / measuredCount_, stageTotalMs_[3] / measuredCount_,
                             stageTotalMs_[4] / measuredCount_, measuredCount_);
                    measuredCount_ = 0;
                    stageTotalMs_.fill(0.0);
                    minRknnMs_ = std::numeric_limits<double>::max();
                    maxRknnMs_ = 0.0;
                }
            }

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
