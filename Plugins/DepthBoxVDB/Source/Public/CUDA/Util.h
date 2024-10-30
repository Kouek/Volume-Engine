#ifndef PUBLIC_DEPTHBOXVDB_UTIL_H
#define PUBLIC_DEPTHBOXVDB_UTIL_H

#include <iostream>
#include <format>
#include <stdexcept>

#include <cuda.h>
#include <cuda_runtime.h>

#include <device_launch_parameters.h>

#include <glm/glm.hpp>

namespace CUDA
{

	inline cudaError_t Check(cudaError_t Err)
	{
		if (Err == cudaSuccess)
			return cudaSuccess;
		auto ErrMsg = std::format("{}: {}\n", cudaGetErrorName(Err), cudaGetErrorString(Err));
		std::cerr << ErrMsg;
		return Err;
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

#ifdef RELEASE
	#define CUDA_CHECK(Call) Call
#else
	#define CUDA_CHECK(Call) CUDA::Check(Call)
#endif

#define CUDA_ALIGN alignas(16)

	constexpr uint32_t ThreadPerBlockX3D = 8;
	constexpr uint32_t ThreadPerBlockY3D = 8;
	constexpr uint32_t ThreadPerBlockZ3D = 8;

	constexpr uint32_t ThreadPerBlockX2D = 16;
	constexpr uint32_t ThreadPerBlockY2D = 16;

} // namespace CUDA

#endif
