#pragma once
#include <iostream>
#include "Block.h"
class Runner
{

public:
	static int run(dim3 gridDim, dim3 blockDim, Thread* p_thread);
};

