#pragma once

#include <optional>
#include <string>
#include <vector>

#include "sherpa-onnx/c-api/c-api.h"


namespace fall_detection
{
    namespace audio
    {
        /**
         * @brief KWS 模型配置
         */
        struct KeywordSpotterConfig
        {
            std::string encoderPath;
            std::string decoderPath;
            std::string joinerPath;

            std::string tokensPath;
            std::string keywordsPath;

            int sampleRate = 16000;
            int featureDim = 80;

            int numThreads = 2;

            int maxActivePaths = 4;
            int numTrailingBlanks = 1;

            float keywordsScore = 1.0f;
            float keywordsThreshold = 0.25f;
        };


        /**
         * @brief sherpa-onnx 离线关键词识别封装
         *
         * 当前职责：
         * 1. 加载 KWS 模型
         * 2. 创建流式识别 Stream
         * 3. 接收 float PCM 音频
         * 4. 返回检测到的关键词
         *
         * 暂时不负责：
         * - ALSA 音频采集
         * - 创建线程
         * - 生成报警事件
         */
        class KeywordSpotter
        {
        public:
            KeywordSpotter() = default;

            ~KeywordSpotter();

            KeywordSpotter(
                const KeywordSpotter&) = delete;

            KeywordSpotter& operator=(
                const KeywordSpotter&) = delete;


            /**
             * @brief 初始化 KWS 模型
             */
            bool initialize(
                const KeywordSpotterConfig& config);


            /**
             * @brief 输入一段归一化后的音频
             *
             * samples 范围：
             * -1.0 ~ 1.0
             *
             * @return
             * 如果检测到关键词，返回关键词字符串；
             * 否则返回 std::nullopt。
             */
            std::optional<std::string>
            processSamples(
                const float* samples,
                int sampleCount);


            /**
             * @brief vector 版本
             */
            std::optional<std::string>
            processSamples(
                const std::vector<float>& samples);


            /**
             * @brief 重置当前 KWS Stream
             */
            void reset();


            bool isInitialized() const
            {
                return initialized_;
            }


        private:
            void release();


        private:
            KeywordSpotterConfig config_;

            const SherpaOnnxKeywordSpotter*
                spotter_ = nullptr;

            const SherpaOnnxOnlineStream*
                stream_ = nullptr;

            bool initialized_ = false;
        };

    } // namespace audio

} // namespace fall_detection