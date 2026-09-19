#pragma once

#include <vector>
#include <chrono>

#include "fall-detection/vision/RKNNInferencer.hpp"

namespace fall_detection
{
    namespace vision
    {
        /**
         * @brief 带临时 Track ID 的人体检测结果
         *
         * trackId 只用于摄像头画面中的短时目标关联，
         * 不代表真实身份，不涉及人脸识别。
         */
        struct TrackedPerson
        {
            int trackId = -1;
            DetectResult detection;
        };


        struct PersonTrackerConfig
        {
            // Bounding Box IoU 达到该值时可认为是同一目标
            float iouThreshold = 0.20f;

            /*
             * 中心点距离阈值。
             *
             * 距离会除以人体框高度进行归一化，
             * 因此不直接依赖摄像头分辨率。
             */
            float centerDistanceThreshold = 0.80f;

            // 人体短暂消失后保留 Track 的时间
            int maxMissingMs = 1500;
        };


        /**
         * @brief 轻量级多人短时跟踪器
         *
         * 使用：
         * Bounding Box IoU
         * +
         * 人体中心距离
         *
         * 对前后帧人体进行关联。
         */
        class PersonTracker
        {
        public:
            explicit PersonTracker(
                const PersonTrackerConfig& config =
                    PersonTrackerConfig());

            /**
             * @brief 输入当前帧检测结果，返回带 Track ID 的人体
             */
            std::vector<TrackedPerson> update(
                const std::vector<DetectResult>& detections);

            /**
             * @brief 清空所有 Track
             */
            void reset();

        private:
            struct Track
            {
                int id = -1;

                DetectResult lastDetection;

                std::chrono::steady_clock::time_point lastSeen;
            };


            float calculateIoU(
                const DetectResult& a,
                const DetectResult& b) const;


            float calculateNormalizedCenterDistance(
                const DetectResult& a,
                const DetectResult& b) const;


            void removeExpiredTracks(
                const std::chrono::steady_clock::time_point& now);


        private:
            PersonTrackerConfig config_;

            std::vector<Track> tracks_;

            int nextTrackId_ = 1;
        };
    }
}