#include "Thread.h"

Thread::Thread():
	m_threadId(), current_part(0)
{
}



const dim3& Thread::getTid() const
{
	return m_threadId;
}

void Thread::updateTid()
{
	threadId = m_threadId;
}
