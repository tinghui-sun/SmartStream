#pragma once

#include <functional>
#include <chrono>
#include <thread>
#include <atomic>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <string>
#include <algorithm>

class TinyTimer
{
public:
	TinyTimer() :m_expired(true), m_tryExpired(false), m_loopExecute(false) {}
	TinyTimer(const TinyTimer& t)
	{
		m_expired = t.m_expired.load();
		m_tryExpired = t.m_tryExpired.load();
	}
	virtual~TinyTimer() { KillTimer(); }

public:
	bool SetTimer(int interval, std::function<void()> task, bool bLoop = false, bool async = true)
	{
		if (!m_expired || m_tryExpired)
			return false;

		m_expired = false;
		m_loopExecute = bLoop;
		m_loopCount = 0;

		// 如果是异步执行
		if (async)
		{
			if (m_thread != nullptr)
				m_thread.reset();
			m_thread = std::make_shared<std::thread>(([this, interval, task]()
			{
#ifdef WIN32
				int _wait = 30, _loop = std::max(1, interval / _wait), _loop1 = 0;
#else
				int _wait = 5, _loop = std::max(1, interval / _wait), _loop1 = 0;
#endif 
				while (!m_tryExpired)
				{
					//std::this_thread::sleep_for(std::chrono::milliseconds(interval));
					std::this_thread::sleep_for(std::chrono::milliseconds(_wait));
					_loop1++;
					if (_loop1 == _loop)
					{
						_loop1 = 0;
						task();
						m_loopCount++;
						if (!m_loopExecute)
							break;
					}
				}
				{
					std::lock_guard<std::mutex> locker(m_threadMutex);
					m_expired = true;
					m_tryExpired = false;
					m_expiredConditionVar.notify_one();
				}
			})
				);
			m_thread->detach();
		}
		else
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(interval));
			if (!m_tryExpired)
				task();
			m_expired = true;
			m_tryExpired = false;
		}
		return true;
	}
	void KillTimer()
	{
		if (m_expired || m_tryExpired || m_thread == nullptr)
			return;
		m_tryExpired = true;
		{
			std::unique_lock<std::mutex> locker(m_threadMutex);
			m_expiredConditionVar.wait(locker, [this] {return m_expired == true; });
			if (m_expired == true)
				m_tryExpired = false;
		}
	}
	template<typename callable, typename... arguments>
	bool AsyncOnceExecute(int interval, callable&& fun, arguments&&... args)
	{
		std::function<typename std::result_of<callable(arguments...)>::type()> task(std::bind(std::forward<callable>(fun), std::forward<arguments>(args)...));
		return SetTimer(interval, task, false);
	}
	template<typename callable, typename... arguments>
	bool AsyncLoopExecute(int interval, callable&& fun, arguments&&... args)
	{
		std::function<typename std::result_of<callable(arguments...)>::type()> task(std::bind(std::forward<callable>(fun), std::forward<arguments>(args)...));
		return SetTimer(interval, task, true);
	}
	template<typename callable, typename... arguments>
	bool SyncOnceExecute(int interval, callable&& fun, arguments&&... args)
	{
		std::function<typename std::result_of<callable(arguments...)>::type()> task(std::bind(std::forward<callable>(fun), std::forward<arguments>(args)...)); //绑定任务函数或lambda成function
		return SetTimer(interval, task, false, false);
	}

private:
	std::atomic_bool m_expired;
	std::atomic_bool m_tryExpired;
	std::atomic_bool m_loopExecute;
	std::mutex m_threadMutex;
	std::condition_variable m_expiredConditionVar;
	std::shared_ptr<std::thread> m_thread;
	unsigned int m_loopCount{ 0 };
};