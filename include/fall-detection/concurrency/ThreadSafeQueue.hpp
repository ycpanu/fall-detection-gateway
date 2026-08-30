#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <chrono>

namespace fall_detection
{
	namespace concurrency
	{
		template<typename T> // C++ 模板语法（泛型），该队列为百搭容器，可以传 cv::Mat 当图像队列，也可以 std::string 当 JSON 报警队列
		class ThreadSafeQueue
		{
		private:
			std::queue<T> queue_;

			//互斥锁 Mutex，必须加 mutable 关键字，因为在 const 函数中我们也需要加锁
			mutable std::mutex mutex_;

			/**
			 * 条件变量
			 * 让一个或多个线程阻塞等待，直到另一个线程改变了某个状态并唤醒它们
			 * 相对于 while(!ready) 死循环的浪费 CPU 资源，条件变量更高效 
			 */
			std::condition_variable cond_var_;

			//队列最大容量，防止摄像头抓图太快把内存占满
			size_t max_size_;

		public:
			//构造函数，默认容量设置为100
			// explicit 防止 C++ 编译器在背后做隐式类型转换
			explicit ThreadSafeQueue(size_t max_size = 100) : max_size_(max_size) {}

			// 1. 压入数据（生产者调用，如视频采集线程不断把新画面塞进来）
			void push(T new_value)
			{
				// lock_guard 是 C++ 的“智能锁”，进该作用域自动把门锁死，出作用域自动开锁。
				// 下面代码抛出异常，锁会自动解开，绝不发生死锁（PAII 机制）
				std::lock_guard<std::mutex> lock(mutex_);

				// 队满自动丢弃最旧的数据
				if (queue_.size() >= max_size_)
				{
					queue_.pop();
				}

				// std::move()：把内存的所有权转移过去，而不是复制一遍
				// 传高清图片时极省内存，速度极快
				queue_.push(std::move(new_value));

				// 唤醒一个正在休眠等待数据的消费者线程
				cond_var_.notify_one();
			}

			// 2. 阻塞获取数据（消费者调用，如 AI 推理线程来拿图片）
			void wait_and_pop(T& value)
			{
				// unique_lock 比 lock_guard 更灵活，允许中途解锁
				// 一般 unique_lock 用于 wait()，lock_guard 用于 notify
				// 因为配合 condition_variable 使用时，线程休眠期间必须把锁还给别人
				std::unique_lock<std::mutex> lock(mutex_);

				// 零 CPU 轮询消耗：如果 queue_ 为空，当前线程就会直接交出 CPU 控制权，进入休眠状态
				// 直到上面 push 函数里的 notify_one() 把它唤醒，它才会继续走下去
				cond_var_.wait(lock,[this]{ return !queue_.empty();});

				// 把队列数据转移给外面的变量
				value = std::move(queue_.front());

				// 弹出已经拿走的数据
				queue_.pop();
			}

			// 3. 尝试获取数据（非阻塞获取，有图拿走返回 True，无图返回 False，绝不死等）
			bool wait_for_and_pop(T& value, std::chrono::milliseconds timeout)
			{
				std::unique_lock<std::mutex> lock(mutex_);

				if (!cond_var_.wait_for(lock, timeout, [this]{ return !queue_.empty(); }))
				{
					return false;
				}

				value = std::move(queue_.front());
				queue_.pop();
				return true;
			}

			bool try_pop(T& value)
			{
				std::lock_guard<std::mutex> lock(mutex_);

				if (queue_.empty())
				{
					return false;
				}
				
				value = std::move(queue_.front());
				queue_.pop();

				return true;
			}

			// 4. 判断队列是否为空
			bool empty() const
			{
				std::lock_guard<std::mutex> lock(mutex_);
				return queue_.empty();
			}

			// 5. 获取当前队列长度
			size_t size() const
			{
				std::lock_guard<std::mutex> lock(mutex_);
				return queue_.size();
			}
		};
	}
}