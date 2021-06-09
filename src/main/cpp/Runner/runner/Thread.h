#pragma once
#include <vector>

#include "PseudoCuda.h"
#include "AbstractMemory.h"

class Thread
{

protected:
	int current_part;
	dim3 m_threadId;

public:
	Thread();

	virtual bool usingShared() = 0;

	const dim3& getTid() const;
	void updateTid();

	virtual int run() = 0;

	virtual AbstractMemory* buildSharedMemory() = 0;
	virtual Thread* build(dim3 threadId, AbstractMemory* shared) { return nullptr; }
};
