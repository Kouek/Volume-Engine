#ifndef DEPTHBOXVDB_D3D12UTIL_H
#define DEPTHBOXVDB_D3D12UTIL_H

#include <cuda.h>
#include <cuda_runtime.h>

#include <d3d12.h>

#include <CUDA/Util.h>

struct D3D12InteropCUDA
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

struct D3D12TextureInteropCUDA : D3D12InteropCUDA
{
	cudaSurfaceObject_t SurfaceObject = 0;
	D3D12_RESOURCE_DESC TextureDesc;

	D3D12TextureInteropCUDA(UINT NodeMask, ID3D12Device* Device, ID3D12Resource* Texture)
	{
		TextureDesc = Texture->GetDesc();

		{
			HANDLE SharedHandle;
			ThrowIfFailed(
				Device->CreateSharedHandle(Texture, NULL, GENERIC_ALL, NULL, &SharedHandle));

			D3D12_RESOURCE_ALLOCATION_INFO D3D12ResourceAllocationInfo;
			D3D12ResourceAllocationInfo =
				Device->GetResourceAllocationInfo(NodeMask, 1, &TextureDesc);

			cudaExternalMemoryHandleDesc ExternalMemoryHandleDesc{};
			ExternalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeD3D12Heap;
			ExternalMemoryHandleDesc.handle.win32.handle = SharedHandle;
			ExternalMemoryHandleDesc.size = D3D12ResourceAllocationInfo.SizeInBytes;
			ExternalMemoryHandleDesc.flags = cudaExternalMemoryDedicated;
			CUDA_CHECK(cudaImportExternalMemory(&ExternalMemory, &ExternalMemoryHandleDesc));
			CloseHandle(SharedHandle);
		}

		cudaExternalMemoryMipmappedArrayDesc ExternalMmeoryMipmappedArrayDesc{};
		ExternalMmeoryMipmappedArrayDesc.extent =
			make_cudaExtent(TextureDesc.Width, TextureDesc.Height, 0);
		switch (TextureDesc.Format)
		{
			case DXGI_FORMAT_R32_FLOAT:
				ExternalMmeoryMipmappedArrayDesc.formatDesc = cudaCreateChannelDesc<float>();
				break;
			case DXGI_FORMAT_R8G8B8A8_UNORM:
				ExternalMmeoryMipmappedArrayDesc.formatDesc = cudaCreateChannelDesc<uchar4>();
				break;
			default:
				assert(false, "Illegal texture format");
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
	}

	~D3D12TextureInteropCUDA()
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

#endif
