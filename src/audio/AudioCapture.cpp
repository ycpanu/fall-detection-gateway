#include "fall-detection/audio/AudioCapture.hpp"
#include "fall-detection/utils/SysLogger.hpp"

#include <cstdint>
#include <utility>


namespace fall_detection
{
    namespace audio
    {
        AudioCapture::~AudioCapture()
        {
            stop();
        }


        bool AudioCapture::start(
            const AudioCaptureConfig& config,
            AudioCallback callback)
        {
            if (running_.load())
            {
                LOG_WARN(
                    "[AudioCapture] 音频采集已经运行"
                );

                return false;
            }


            config_ = config;
            callback_ = std::move(callback);


            if (!callback_)
            {
                LOG_ERROR(
                    "[AudioCapture] AudioCallback 为空"
                );

                return false;
            }


            if (!openDevice())
            {
                return false;
            }


            running_.store(true);


            captureThread_ =
                std::thread(
                    &AudioCapture::captureLoop,
                    this
                );


            LOG_INFO(
                "[AudioCapture] 启动成功："
                "device={}, sampleRate={}, chunk={}",
                config_.device,
                config_.sampleRate,
                config_.framesPerChunk
            );


            return true;
        }


        void AudioCapture::stop()
        {
            if (!running_.exchange(false))
            {
                closeDevice();
                return;
            }


            if (captureThread_.joinable())
            {
                captureThread_.join();
            }


            closeDevice();


            LOG_INFO(
                "[AudioCapture] 音频采集已停止"
            );
        }


        bool AudioCapture::openDevice()
        {
            LOG_INFO("[AudioCapture] 准备打开音频设备: {}", config_.device);
            int ret =
                snd_pcm_open(
                    &pcmHandle_,
                    config_.device.c_str(),
                    SND_PCM_STREAM_CAPTURE,
                    0
                );


            if (ret < 0)
            {
                LOG_ERROR(
                    "[AudioCapture] 打开麦克风失败：{}",
                    snd_strerror(ret)
                );

                pcmHandle_ = nullptr;

                return false;
            }

            LOG_INFO("[AudioCapture] snd_pcm_open成功: {}", config_.device);


            /*
             * 设置：
             *
             * S16_LE
             * 单声道
             * 16000 Hz
             * interleaved
             *
             * soft_resample = 1：
             * 允许 ALSA plug 层进行必要格式转换。
             *
             * latency = 100000 us：
             * 约100ms。
             */
            ret =
                snd_pcm_set_params(
                    pcmHandle_,
                    SND_PCM_FORMAT_S16_LE,
                    SND_PCM_ACCESS_RW_INTERLEAVED,
                    config_.channels,
                    config_.sampleRate,
                    1,
                    100000
                );


            if (ret < 0)
            {
                LOG_ERROR(
                    "[AudioCapture] 配置麦克风失败：{}",
                    snd_strerror(ret)
                );

                closeDevice();

                return false;
            }


            return true;
        }


        void AudioCapture::closeDevice()
        {
            if (pcmHandle_ != nullptr)
            {
                snd_pcm_close(
                    pcmHandle_
                );

                pcmHandle_ = nullptr;
            }
        }


        void AudioCapture::captureLoop()
        {
            /*
             * ALSA 输入：
             *
             * S16_LE
             * -32768 ~ 32767
             */
            std::vector<int16_t> pcmBuffer(
                config_.framesPerChunk *
                config_.channels
            );


            /*
             * sherpa-onnx 输入：
             *
             * float
             * -1.0 ~ 1.0
             */
            std::vector<float> floatBuffer(
                config_.framesPerChunk
            );


            while (running_.load())
            {
                const snd_pcm_sframes_t frames =
                    snd_pcm_readi(
                        pcmHandle_,
                        pcmBuffer.data(),
                        config_.framesPerChunk
                    );


                if (frames == -EPIPE)
                {
                    /*
                     * ALSA XRUN：
                     * 程序来不及读取导致缓冲区溢出。
                     */
                    LOG_WARN(
                        "[AudioCapture] ALSA XRUN，重新准备设备"
                    );

                    snd_pcm_prepare(
                        pcmHandle_
                    );

                    continue;
                }


                if (frames < 0)
                {
                    const int recoverResult =
                        snd_pcm_recover(
                            pcmHandle_,
                            static_cast<int>(frames),
                            1
                        );


                    if (recoverResult < 0)
                    {
                        LOG_ERROR(
                            "[AudioCapture] 音频采集错误：{}",
                            snd_strerror(
                                recoverResult
                            )
                        );
                    }

                    continue;
                }


                if (frames == 0)
                {
                    continue;
                }


                floatBuffer.resize(
                    static_cast<std::size_t>(frames)
                );


                /*
                 * 当前项目固定单声道。
                 *
                 * int16：
                 * -32768 ~ 32767
                 *
                 * 转换：
                 * -1.0 ~ 1.0
                 */
                for (snd_pcm_sframes_t i = 0;
                     i < frames;
                     ++i)
                {
                    floatBuffer[
                        static_cast<std::size_t>(i)
                    ] =
                        static_cast<float>(
                            pcmBuffer[
                                static_cast<std::size_t>(i)
                            ]
                        )
                        / 32768.0f;
                }


                /*
                 * 把当前100ms左右音频交给上层。
                 *
                 * AudioCapture 本身完全不知道
                 * “救命”是什么。
                 */
                callback_(
                    floatBuffer
                );
            }
        }

    } // namespace audio
} // namespace fall_detection