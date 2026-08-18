#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "rknn_api.h"

namespace fall_detection
{
    /**
     *  @brief 目标检测结果结构体
     *  记录 NPU 推理输出的单个人体姿态和边界框信息
     */
    struct DetectResult
    {
        int classId;            // 类别ID(0:stand, 1:sit, 2:bend, 3:lie)
        float confidence;       // 置信度
        int x;                  // 边界框左上角 X 坐标
        int y;                  // 左上角 Y 坐标
        int width;              // 边界框宽度
        int height;             //
    };

    /**
     *  @brief 硬件加速推理类
     * 
     *  封装瑞芯微 RKNN C API，负责将模型加载到开发板 NPU 中，
     *  并对 CameraStreamer 传来的图片进行极速推理。
     */
    class RKNNInferencer
    {
        public:
            /**
             * @param modelPath 转换好的 .rknn 模型文件在开发板的绝对路径
             */
            RKNNInferencer(const std::string& modelPath);
            
            ~RKNNInferencer();

            bool init();

            /**
             * @brief 执行一帧图像的硬件加速推理
             * @param frame 从队列中取出的 OpenCV 原始图像
             * @param results 引用传递，用于接收解析后的检测结果列表
             * @return 推理是否成功
             */
            bool detect(const cv::Mat& frame, std::vector<DetectResult>& results);

        private:
            std::string modelPath_;     //模型文件路径
            rknn_context ctx_;          // RKNN 运行上下文句柄
            bool isInitialized_;        // 模型是否初始化

            // 模型的输入/输出属性
            rknn_tensor_attr* inputAttrs_;
            rknn_tensor_attr* outputAttrs_;
            int numInput_;
            int numOutput_;

            // YOLOv8 训练模型后处理阈值
            const float CONF_THRESHOLD = 0.50f; // 置信度过滤阈值
            const float NMS_THRESHOLD = 0.45f;  // NMS 交并比剔除阈值
            const int NUM_CLASSES = 4;          // 姿态类别数 (stand, sit, bend, lie)

        private:
            /**
             *  @brief 内部辅助函数：读取 .rknn 模型文件到内存
             */
            unsigned char* loadModelFile(const char* filename, int* modelSize);

            // 执行NMS算法，剔除重叠的废框
            void nms(std::vector<DetectResult>& inputBoxes, std::vector<DetectResult>& outputBoxes);

            // 计算两个矩形框的 IoU(交并比)
            float calculateIoU(const DetectResult& box1, const DetectResult& box2);
    };
}