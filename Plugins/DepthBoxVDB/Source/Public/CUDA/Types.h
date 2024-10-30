#ifndef PUBLIC_CUDA_TYPES_H
#define PUBLIC_CUDA_TYPES_H

#include <iostream>
#include <format>
#define _HAS_CXX17 1 // NVCC doesn't define this macro, which is needed in IntelliSense
#include <optional>

#include <vector>

#include <glm/glm.hpp>

#include <CUDA/Util.h>

namespace CUDA
{
	struct Noncopyable
	{
		Noncopyable() = default;
		Noncopyable(const Noncopyable&) = delete;
		Noncopyable& operator=(const Noncopyable&) = delete;
	};

	class CompletenessCheckable
	{
	public:
		CompletenessCheckable(bool bInitialIsComplete) : bIsComplete(bInitialIsComplete) {}

		bool IsComplete() const { return bIsComplete; }

	protected:
		bool bIsComplete;
	};

	class Array : Noncopyable, public CompletenessCheckable
	{
	private:
		cudaArray_t Data = nullptr;

	public:
		Array(const cudaChannelFormatDesc& ChannelDesc, const glm::vec<3, size_t>& Dimension)
			: CompletenessCheckable(true)
		{
			if (Dimension.z != 0)
			{
				auto extent = make_cudaExtent(Dimension.x, Dimension.y, Dimension.z);
				bIsComplete =
					cudaSuccess == CUDA_CHECK(cudaMalloc3DArray(&Data, &ChannelDesc, extent));
			}
			else
				bIsComplete = cudaSuccess
					== CUDA_CHECK(cudaMallocArray(&Data, &ChannelDesc, Dimension.x, Dimension.y));

			if (Data)
			{
				std::cout << std::format("New CUDA::Array {:x}\n", (uint64_t)Data);
			}
		}
		template <typename T>
		Array(const T* InData, const glm::vec<3, size_t>& Dimension,
			const std::optional<cudaChannelFormatDesc>& ChannelDescOpt = {})
			: CompletenessCheckable(true)
		{
			auto chnDesc = ChannelDescOpt.value_or(cudaCreateChannelDesc<T>());
			if (Dimension.z != 0)
			{
				auto Extent = make_cudaExtent(Dimension.x, Dimension.y, Dimension.z);
				bIsComplete = cudaSuccess == CUDA_CHECK(cudaMalloc3DArray(&Data, &chnDesc, Extent));
				if (!bIsComplete)
					return;

				cudaMemcpy3DParms Params{};
				Params.srcPtr =
					make_cudaPitchedPtr(InData, sizeof(T) * Dimension.x, Dimension.x, Dimension.y);
				Params.extent = Extent;
				Params.dstArray = Data;
				Params.kind = cudaMemcpyHostToDevice;
				bIsComplete &= cudaSuccess == CUDA_CHECK(cudaMemcpy3D(&Params));
			}
			else
			{
				bIsComplete = cudaSuccess
					== CUDA_CHECK(cudaMallocArray(&Data, &chnDesc, Dimension.x, Dimension.y));
				if (!bIsComplete)
					return;

				bIsComplete &= cudaSuccess
					== CUDA_CHECK(cudaMemcpyToArray(Data, 0, 0, InData,
						sizeof(T) * Dimension.x * std::max(size_t(1), Dimension.y),
						cudaMemcpyHostToDevice));
			}

			if (Data)
			{
				std::cout << std::format("New CUDA::Array {:x}\n", (uint64_t)Data);
			}
		}
		~Array()
		{
			if (Data)
			{
				std::cout << std::format("Free CUDA::Array {:x}\n", (uint64_t)Data);
				CUDA_CHECK(cudaFreeArray(Data));
			}
		}

		cudaArray_t Get() const { return Data; }
		cudaExtent	GetExtent() const
		{
			cudaChannelFormatDesc ChannelDesc;
			cudaExtent			  Extent;
			CUDA_CHECK(cudaArrayGetInfo(&ChannelDesc, &Extent, nullptr, Data));

			return Extent;
		}
	};

	class Texture : Noncopyable, public CompletenessCheckable
	{
	private:
		cudaTextureObject_t Data = 0;

		std::shared_ptr<Array> Arr;

	public:
		Texture(std::shared_ptr<Array> InArr, const std::optional<cudaTextureDesc>& TexDescOpt = {})
			: CompletenessCheckable(true)
		{
			cudaTextureDesc TexDesc{};
			if (TexDescOpt.has_value())
				TexDesc = TexDescOpt.value();
			else
			{
				TexDesc.normalizedCoords = 0;
				TexDesc.filterMode = cudaFilterModeLinear;
				TexDesc.addressMode[0] = TexDesc.addressMode[1] = TexDesc.addressMode[2] =
					cudaAddressModeBorder;
				TexDesc.readMode = cudaReadModeNormalizedFloat;
			}

			Arr = InArr;

			cudaResourceDesc ResDesc{};
			ResDesc.resType = cudaResourceTypeArray;
			ResDesc.res.array.array = Arr->Get();

			bIsComplete = cudaSuccess
				== CUDA_CHECK(cudaCreateTextureObject(&Data, &ResDesc, &TexDesc, nullptr));

			if (Data != 0)
			{
				std::cout << std::format("New CUDA::Texture {}\n", Data);
			}
		}
		~Texture()
		{
			if (Data != 0)
			{
				std::cout << std::format("Free CUDA::Texture {}\n", Data);
				CUDA_CHECK(cudaDestroyTextureObject(Data));
			}
		}

		cudaTextureObject_t Get() const { return Data; }
		const Array*		GetArray() const { return Arr.get(); }
		cudaTextureDesc		GetDesc() const
		{
			cudaTextureDesc Ret;
			CUDA_CHECK(cudaGetTextureObjectTextureDesc(&Ret, Data));
			return Ret;
		}
	};

	class Surface : Noncopyable, public CompletenessCheckable
	{
	private:
		cudaSurfaceObject_t Data = 0;

		std::shared_ptr<Array> Arr;

	public:
		Surface(std::shared_ptr<Array> InArr) : CompletenessCheckable(true)
		{
			Arr = InArr;

			cudaResourceDesc ResDesc{};
			ResDesc.resType = cudaResourceTypeArray;
			ResDesc.res.array.array = Arr->Get();

			bIsComplete = cudaSuccess == CUDA_CHECK(cudaCreateSurfaceObject(&Data, &ResDesc));

			if (Data != 0)
			{
				std::cout << std::format("New CUDA::Surface {}\n", Data);
			}
		}
		~Surface()
		{
			if (Data != 0)
			{
				std::cout << std::format("Free CUDA::Surface {}\n", Data);
				CUDA_CHECK(cudaDestroySurfaceObject(Data));
			}
		}

		cudaSurfaceObject_t Get() const { return Data; }
	};

} // namespace CUDA

#endif // !PUBLIC_CUDA_TYPES_H
