#include "Block.h"

Block::Block(dim3 blockId, Thread* p_thread):
	threads(), logs(), m_blockId(blockId)
{
	shared = p_thread->buildSharedMemory();
	size_t threads_number = blockDim.x * blockDim.y * blockDim.z;
	dim3 tid;
	threads.reserve(threads_number);
	for (size_t i = 0; i < threads_number; i++) {
		if (tid.x == blockDim.x) {
			tid.x = 0;
			tid.y++;
			if (tid.y == blockDim.y) {
				tid.y = 0;
				tid.z++;
			}
		}
		threads.push_back(p_thread->build(tid, shared));
		tid.x++;
	}
}

Block::~Block()
{
	for (Thread* t : threads) delete t;
	delete shared;
}

int Block::run()
{
	blockId = m_blockId;
	bool ended = false;
	while (!ended) {
		ended = true;
		int c = 0;
		for (Thread* t : threads) {
			threadId = t->getTid();
			int res = t->run();
			if (res == -1) {
				logs += "Error with thread " + t->getTid() + "!\n";
				return -1;
			}
			if (res == 0) {
				if (c > 0 && !ended) {
					logs += "Thread " + t->getTid() + " ended earlier!\n";
					return -1;
				}
			}
			else {
				ended = false;
			}
			c++;
		}
	}
	return 0;
}

const std::string Block::getLogs() const
{
	return "Block " + blockId + " logs:\n" + logs;
}
