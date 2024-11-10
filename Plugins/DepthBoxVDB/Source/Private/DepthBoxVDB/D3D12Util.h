#ifndef DEPTHBOXVDB_D3D12UTIL_H
#define DEPTHBOXVDB_D3D12UTIL_H

#include <cuda.h>
#include <cuda_runtime.h>

#include <d3d12.h>

#include <CUDA/Util.h>

namespace DepthBoxVDB
{
	namespace D3D12
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

		struct MappedCUDAResource : Noncopyable
		{
			cudaExternalMemory_t	ExternalMemory = nullptr;
			cudaExternalSemaphore_t ExternalSemaphore = nullptr;
		};

		inline std::string HrToString(HRESULT Hr)
		{
			char Str[64] = {};
			sprintf_s(Str, "HRESULT of 0x%08X", static_cast<UINT>(Hr));
			return std::string(Str);
		}

		inline void ThrowIfFailed(HRESULT Hr)
		{
			if (FAILED(Hr))
			{
				auto ErrMsg = HrToString(Hr);
				std::cerr << ErrMsg;
				throw std::runtime_error(ErrMsg);
			}
		}

		struct TextureMappedCUDASurface : MappedCUDAResource, CompletenessCheckable
		{
			cudaSurfaceObject_t SurfaceObject = 0;
			D3D12_RESOURCE_DESC TextureDesc;

			TextureMappedCUDASurface(UINT NodeMask, ID3D12Device* Device, ID3D12Resource* Texture)
				: CompletenessCheckable(false)
			{
				if (!Texture)
					return;

				TextureDesc = Texture->GetDesc();

				{
					HANDLE SharedHandle;
					ThrowIfFailed(Device->CreateSharedHandle(
						Texture, NULL, GENERIC_ALL, NULL, &SharedHandle));

					D3D12_RESOURCE_ALLOCATION_INFO D3D12ResourceAllocationInfo;
					D3D12ResourceAllocationInfo =
						Device->GetResourceAllocationInfo(NodeMask, 1, &TextureDesc);

					cudaExternalMemoryHandleDesc ExternalMemoryHandleDesc{};
					ExternalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeD3D12Heap;
					ExternalMemoryHandleDesc.handle.win32.handle = SharedHandle;
					ExternalMemoryHandleDesc.size = D3D12ResourceAllocationInfo.SizeInBytes;
					ExternalMemoryHandleDesc.flags = cudaExternalMemoryDedicated;
					CUDA_CHECK(
						cudaImportExternalMemory(&ExternalMemory, &ExternalMemoryHandleDesc));
					CloseHandle(SharedHandle);
				}

				cudaExternalMemoryMipmappedArrayDesc ExternalMmeoryMipmappedArrayDesc{};
				ExternalMmeoryMipmappedArrayDesc.extent =
					make_cudaExtent(TextureDesc.Width, TextureDesc.Height, 0);
				switch (TextureDesc.Format)
				{
					case DXGI_FORMAT_R32_FLOAT:
						ExternalMmeoryMipmappedArrayDesc.formatDesc =
							cudaCreateChannelDesc<float>();
						break;
					case DXGI_FORMAT_R8G8B8A8_UNORM:
						ExternalMmeoryMipmappedArrayDesc.formatDesc =
							cudaCreateChannelDesc<uchar4>();
						break;
					default:
						throw std::runtime_error("Illegal texture format");
				}
				ExternalMmeoryMipmappedArrayDesc.numLevels = 1;
				ExternalMmeoryMipmappedArrayDesc.flags = cudaArraySurfaceLoadStore;

				cudaMipmappedArray_t MipmappedArray = nullptr;
				CUDA_CHECK(cudaExternalMemoryGetMappedMipmappedArray(
					&MipmappedArray, ExternalMemory, &ExternalMmeoryMipmappedArrayDesc));

				cudaArray_t Array = nullptr;
				CUDA_CHECK(cudaGetMipmappedArrayLevel(&Array, MipmappedArray, 0));

				cudaResourceDesc ResDesc{};
				ResDesc.resType = cudaResourceTypeArray;
				ResDesc.res.array.array = Array;
				CUDA_CHECK(cudaCreateSurfaceObject(&SurfaceObject, &ResDesc));

				bIsComplete = true;
			}

			~TextureMappedCUDASurface()
			{
				if (SurfaceObject != 0)
				{
					CUDA_CHECK(cudaDestroySurfaceObject(SurfaceObject));
				}
				if (!ExternalMemory)
				{
					CUDA_CHECK(cudaDestroyExternalMemory(ExternalMemory));
				}
			}
		};

		struct TextureMappedCUDATexture : MappedCUDAResource, CompletenessCheckable
		{
			cudaTextureObject_t TextureObject = 0;
			D3D12_RESOURCE_DESC TextureDesc;

			TextureMappedCUDATexture(UINT NodeMask, ID3D12Device* Device, ID3D12Resource* Texture)
				: CompletenessCheckable(false)
			{
				if (!Texture)
					return;

				TextureDesc = Texture->GetDesc();

				{
					HANDLE SharedHandle;
					ThrowIfFailed(Device->CreateSharedHandle(
						Texture, NULL, GENERIC_ALL, NULL, &SharedHandle));

					D3D12_RESOURCE_ALLOCATION_INFO D3D12ResourceAllocationInfo;
					D3D12ResourceAllocationInfo =
						Device->GetResourceAllocationInfo(NodeMask, 1, &TextureDesc);

					cudaExternalMemoryHandleDesc ExternalMemoryHandleDesc{};
					ExternalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeD3D12Heap;
					ExternalMemoryHandleDesc.handle.win32.handle = SharedHandle;
					ExternalMemoryHandleDesc.size = D3D12ResourceAllocationInfo.SizeInBytes;
					ExternalMemoryHandleDesc.flags = cudaExternalMemoryDedicated;
					CUDA_CHECK(
						cudaImportExternalMemory(&ExternalMemory, &ExternalMemoryHandleDesc));
					CloseHandle(SharedHandle);
				}

				cudaExternalMemoryMipmappedArrayDesc ExternalMmeoryMipmappedArrayDesc{};
				ExternalMmeoryMipmappedArrayDesc.extent =
					make_cudaExtent(TextureDesc.Width, TextureDesc.Height, 0);
				switch (TextureDesc.Format)
				{
					case DXGI_FORMAT_R32_FLOAT:
						ExternalMmeoryMipmappedArrayDesc.formatDesc =
							cudaCreateChannelDesc<float>();
						break;
					case DXGI_FORMAT_R8G8B8A8_UNORM:
						ExternalMmeoryMipmappedArrayDesc.formatDesc =
							cudaCreateChannelDesc<uchar4>();
						break;
					default:
						throw std::runtime_error("Illegal texture format");
				}
				ExternalMmeoryMipmappedArrayDesc.numLevels = 1;
				ExternalMmeoryMipmappedArrayDesc.flags = cudaArraySurfaceLoadStore;

				cudaMipmappedArray_t MipmappedArray = nullptr;
				CUDA_CHECK(cudaExternalMemoryGetMappedMipmappedArray(
					&MipmappedArray, ExternalMemory, &ExternalMmeoryMipmappedArrayDesc));

				cudaArray_t Array = nullptr;
				CUDA_CHECK(cudaGetMipmappedArrayLevel(&Array, MipmappedArray, 0));

				cudaResourceDesc ResDesc{};
				ResDesc.resType = cudaResourceTypeArray;
				ResDesc.res.array.array = Array;

				cudaTextureDesc CUDATexDesc{};
				CUDATexDesc.normalizedCoords = 1;
				CUDATexDesc.filterMode = cudaFilterModeLinear;
				CUDATexDesc.addressMode[0] = CUDATexDesc.addressMode[1] =
					CUDATexDesc.addressMode[2] = cudaAddressModeBorder;
				CUDATexDesc.readMode = cudaReadModeNormalizedFloat;
				CUDA_CHECK(
					cudaCreateTextureObject(&TextureObject, &ResDesc, &CUDATexDesc, nullptr));

				bIsComplete = true;
			}

			~TextureMappedCUDATexture()
			{
				if (TextureObject != 0)
				{
					CUDA_CHECK(cudaDestroySurfaceObject(TextureObject));
				}
				if (!ExternalMemory)
				{
					CUDA_CHECK(cudaDestroyExternalMemory(ExternalMemory));
				}
			}
		};

	} // namespace D3D12
} // namespace DepthBoxVDB

#endif
