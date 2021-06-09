#include "Runner.h"

int Runner::run(dim3 griddim, dim3 blockdim, Thread* p_thread)
{
	gridDim = griddim;
	blockDim = blockdim;
	size_t number_of_blocks = gridDim.x * gridDim.y * gridDim.z;
	dim3 bid;
	for (size_t i = 0; i < number_of_blocks; i++) {
		if (bid.x == gridDim.x) {
			bid.x = 0;
			bid.y++;
			if (bid.y == gridDim.y) {
				bid.y = 0;
				bid.z++;
			}
		}
		Block block(bid, p_thread);
		int res = block.run();
		if (res != 0) {
			std::cout << block.getLogs();
			return -1;
		}
		bid.x++;
	}
	return 0;
}
