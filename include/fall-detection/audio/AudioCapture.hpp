#pragma once

#include <alsa/asoundlib.h>

#include <atomic>
#include <functional>
#include <string>
#include <thread>
#include <vector>

namespace fall_detection
{
    namespace audio
    {
        struct AudioCaptureConfig
        {
            // 目前开发板确认的 USB 麦克风
            std::string device = "plughw:CARD=Device,DEV=0";

            int sampleRate = 16000;
            int channels = 1;

            // 每次读取 100ms：
            // 16000 × 0.1 = 1600 samples
            int framesPerChunk = 1600;
        };


        class AudioCapture
        {
        public:
            using AudioCallback =
                std::function<void(
                    const std::vector<float>& samples)>;


            AudioCapture() = default;

            ~AudioCapture();


            AudioCapture(
                const AudioCapture&) = delete;

            AudioCapture& operator=(
                const AudioCapture&) = delete;


            bool start(
                const AudioCaptureConfig& config,
                AudioCallback callback);


            void stop();


            bool isRunning() const
            {
                return running_.load();
            }


        private:
            bool openDevice();

            void closeDevice();

            void captureLoop();


        private:
            AudioCaptureConfig config_;

            AudioCallback callback_;

            snd_pcm_t* pcmHandle_ = nullptr;

            std::thread captureThread_;

            std::atomic<bool> running_{false};
        };

    } // namespace audio
} // namespace fall_detection