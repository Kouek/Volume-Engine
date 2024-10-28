#ifndef PUBLIC_DEPTHBOXVDB_UTIL_H
#define PUBLIC_DEPTHBOXVDB_UTIL_H

#include <iostream>
#include <format>
#include <stdexcept>

#include <cuda.h>
#include <cuda_runtime.h>

#include <device_launch_parameters.h>

#include <glm/glm.hpp>

namespace DepthBoxVDB
{
	inline void ThrowIfFailed(cudaError_t Err)
	{
		if (Err == cudaSuccess)
			return;
		auto ErrMsg = std::format("{}: {}\n", cudaGetErrorName(Err), cudaGetErrorString(Err));
		std::cerr << ErrMsg;
		throw std::runtime_error(ErrMsg);
	}

#ifdef __CUDACC__
	template <typename FuncType> __global__ void ParallelKernel(FuncType Func)
	{
		glm::uvec3 DispatchThreadID(blockIdx.x * blockDim.x + threadIdx.x,
			blockIdx.y * blockDim.y + threadIdx.y, blockIdx.z * blockDim.z + threadIdx.z);
		Func(DispatchThreadID);
	}
#endif

	template <typename FuncType>
	void ParallelFor(const dim3 BlockPerGrid, const dim3 ThreadPerBlock, FuncType Func,
		cudaStream_t Stream = nullptr)
	{
#ifdef __CUDACC__
		ParallelKernel<<<BlockPerGrid, ThreadPerBlock, 0, Stream>>>(Func);
#endif
	}

} // namespace DepthBoxVDB

#ifdef RELEASE
	#define DEPTHBOXVDB_CHECK(Call) Call
#else
	#define DEPTHBOXVDB_CHECK(Call) DepthBoxVDB::ThrowIfFailed(Call)
#endif

#define DEPTHBOXVDB_ALIGN alignas(16)

#endif // !PUBLIC_CUDA_UTIL_H
