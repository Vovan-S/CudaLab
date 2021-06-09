#pragma once
#include "Thread.h"
#include <string>

class Block
{
	std::vector<Thread*> threads;
	AbstractMemory* shared;
	std::string logs;
	dim3 m_blockId;

public:
	Block(dim3 blockId, Thread* p_thread);
	~Block();
	int run();
	const std::string getLogs() const;
};

