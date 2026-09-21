#include "fall-detection/audio/KeywordSpotter.hpp"
#include "fall-detection/utils/SysLogger.hpp"

#include <cstring>


namespace fall_detection
{
    namespace audio
    {
        KeywordSpotter::~KeywordSpotter()
        {
            release();
        }


        bool KeywordSpotter::initialize(
            const KeywordSpotterConfig& config)
        {
            release();

            config_ = config;


            SherpaOnnxKeywordSpotterConfig sherpaConfig;

            std::memset(
                &sherpaConfig,
                0,
                sizeof(sherpaConfig)
            );


            // ==========================================
            // 音频特征配置
            // ==========================================

            sherpaConfig.feat_config.sample_rate =
                config_.sampleRate;

            sherpaConfig.feat_config.feature_dim =
                config_.featureDim;


            // ==========================================
            // KWS 模型
            // ==========================================

            sherpaConfig
                .model_config
                .transducer
                .encoder =
                    config_.encoderPath.c_str();

            sherpaConfig
                .model_config
                .transducer
                .decoder =
                    config_.decoderPath.c_str();

            sherpaConfig
                .model_config
                .transducer
                .joiner =
                    config_.joinerPath.c_str();


            sherpaConfig.model_config.tokens =
                config_.tokensPath.c_str();


            // 在 Orange Pi CPU 上运行
            sherpaConfig.model_config.provider =
                "cpu";

            sherpaConfig.model_config.num_threads =
                config_.numThreads;

            sherpaConfig.model_config.debug =
                0;


            // ==========================================
            // 关键词配置
            // ==========================================

            sherpaConfig.keywords_file =
                config_.keywordsPath.c_str();

            sherpaConfig.max_active_paths =
                config_.maxActivePaths;

            sherpaConfig.num_trailing_blanks =
                config_.numTrailingBlanks;

            sherpaConfig.keywords_score =
                config_.keywordsScore;

            sherpaConfig.keywords_threshold =
                config_.keywordsThreshold;


            // ==========================================
            // 创建 KWS
            // ==========================================

            spotter_ =
                SherpaOnnxCreateKeywordSpotter(
                    &sherpaConfig
                );


            if (spotter_ == nullptr)
            {
                LOG_ERROR(
                    "[KeywordSpotter] "
                    "创建 sherpa-onnx KWS 失败"
                );

                release();

                return false;
            }


            stream_ =
                SherpaOnnxCreateKeywordStream(
                    spotter_
                );


            if (stream_ == nullptr)
            {
                LOG_ERROR(
                    "[KeywordSpotter] "
                    "创建 KWS Stream 失败"
                );

                release();

                return false;
            }


            initialized_ = true;


            LOG_INFO(
                "[KeywordSpotter] 初始化成功："
                "sampleRate={}, threads={}, "
                "keywords={}",
                config_.sampleRate,
                config_.numThreads,
                config_.keywordsPath
            );


            return true;
        }


        std::optional<std::string>
        KeywordSpotter::processSamples(
            const float* samples,
            int sampleCount)
        {
            if (!initialized_ ||
                samples == nullptr ||
                sampleCount <= 0)
            {
                return std::nullopt;
            }


            /*
             * sherpa-onnx 要求输入 float PCM，
             * 样本范围为 -1.0 ~ 1.0。
             */
            SherpaOnnxOnlineStreamAcceptWaveform(
                stream_,
                config_.sampleRate,
                samples,
                sampleCount
            );


            /*
             * 一次输入音频后，
             * 可能已经积累了多帧可以进行推理，
             * 所以这里需要 while，而不是 if。
             */
            while (
                SherpaOnnxIsKeywordStreamReady(
                    spotter_,
                    stream_
                )
            )
            {
                SherpaOnnxDecodeKeywordStream(
                    spotter_,
                    stream_
                );


                const SherpaOnnxKeywordResult* result =
                    SherpaOnnxGetKeywordResult(
                        spotter_,
                        stream_
                    );


                if (result == nullptr)
                {
                    continue;
                }


                std::optional<std::string>
                    detectedKeyword;


                if (result->keyword != nullptr &&
                    std::strlen(
                        result->keyword
                    ) > 0)
                {
                    detectedKeyword =
                        std::string(
                            result->keyword
                        );


                    LOG_INFO(
                        "[KeywordSpotter] "
                        "检测到关键词：{}",
                        *detectedKeyword
                    );


                    /*
                     * sherpa-onnx 官方要求：
                     * 检测到关键词后立即 Reset。
                     */
                    SherpaOnnxResetKeywordStream(
                        spotter_,
                        stream_
                    );
                }


                /*
                 * GetKeywordResult 返回的对象
                 * 必须释放。
                 */
                SherpaOnnxDestroyKeywordResult(
                    result
                );


                if (detectedKeyword)
                {
                    return detectedKeyword;
                }
            }


            return std::nullopt;
        }


        std::optional<std::string>
        KeywordSpotter::processSamples(
            const std::vector<float>& samples)
        {
            if (samples.empty())
            {
                return std::nullopt;
            }


            return processSamples(
                samples.data(),
                static_cast<int>(
                    samples.size()
                )
            );
        }


        void KeywordSpotter::reset()
        {
            if (spotter_ != nullptr &&
                stream_ != nullptr)
            {
                SherpaOnnxResetKeywordStream(
                    spotter_,
                    stream_
                );
            }
        }


        void KeywordSpotter::release()
        {
            /*
             * Stream 必须先销毁，
             * 再销毁 KeywordSpotter。
             */
            if (stream_ != nullptr)
            {
                SherpaOnnxDestroyOnlineStream(
                    stream_
                );

                stream_ = nullptr;
            }


            if (spotter_ != nullptr)
            {
                SherpaOnnxDestroyKeywordSpotter(
                    spotter_
                );

                spotter_ = nullptr;
            }


            initialized_ = false;
        }

    } // namespace audio

} // namespace fall_detection