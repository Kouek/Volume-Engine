#include "VolRenderer.h"

DepthBoxVDB::D3D12::RendererSharedStates& DepthBoxVDB::D3D12::RendererSharedStates::Instance()
{
	static DepthBoxVDB::D3D12::RendererSharedStates State;
	return State;
}

void DepthBoxVDB::D3D12::RendererSharedStates::Register(void* Device, void* ExternalTexture)
{
	if (!ExternalTexture)
		return;

	ID3D12Device*	DeviceNative = reinterpret_cast<ID3D12Device*>(Device);
	ID3D12Resource* ExternalTextureNative = reinterpret_cast<ID3D12Resource*>(ExternalTexture);

	auto RegisteredTexture = std::make_shared<D3D12::TextureMappedCUDASurface>(
		D3D12NodeMask, DeviceNative, ExternalTextureNative);
	ExternalToRegisteredTextures.emplace(ExternalTexture, RegisteredTexture);
}

void DepthBoxVDB::D3D12::RendererSharedStates::Unregister(void* RegisteredTexture)
{
	if (!RegisteredTexture)
		return;

	auto ItrDel = ExternalToRegisteredTextures.end();
	for (auto Itr = ExternalToRegisteredTextures.begin(); Itr != ExternalToRegisteredTextures.end();
		 ++Itr)
	{
		if (Itr->second.get() == RegisteredTexture)
		{
			ItrDel = Itr;
			break;
		}
	}
	if (ItrDel != ExternalToRegisteredTextures.end())
		return;

	ExternalToRegisteredTextures.erase(ItrDel);
}

DepthBoxVDB::D3D12::RendererSharedStates::RendererSharedStates()
{
	int DeviceNum = 0;
	CUDA_CHECK(cudaGetDeviceCount(&DeviceNum));
	assert(DeviceNum > 0);

	cudaDeviceProp Prop;
	CUDA_CHECK(cudaGetDeviceProperties(&Prop, 0));
	D3D12NodeMask = Prop.luidDeviceNodeMask;

	CUDA_CHECK(cudaStreamCreateWithFlags(&Stream, cudaStreamNonBlocking));
}

DepthBoxVDB::D3D12::RendererSharedStates::~RendererSharedStates()
{
	if (Stream != 0)
	{
		CUDA_CHECK(cudaStreamDestroy(Stream));
	}
}

DepthBoxVDB::VolRenderer::Renderer::Renderer(const CreateParameters& Params)
	: RHIType(Params.RHIType)
{
	switch (RHIType)
	{
		case ERHIType::D3D12:
			Stream = D3D12::RendererSharedStates::Instance().Stream;
			break;
		default:
			assert(false);
	}
}

DepthBoxVDB::VolRenderer::Renderer::~Renderer() {}

void DepthBoxVDB::VolRenderer::Renderer::Register(const RegisterParameters& Params)
{
	switch (RHIType)
	{
		case DepthBoxVDB::VolRenderer::ERHIType::D3D12:
			for (auto ExternalTexture : { Params.InSceneDepthTexture, Params.InOutColorTexture })
				D3D12::RendererSharedStates::Instance().Register(Params.Device, ExternalTexture);
			break;
		default:
			assert(false);
	}

	auto FindAndSet = [&](std::shared_ptr<D3D12::TextureMappedCUDASurface>& Dest,
						  void*												ExternalTexture) {
		auto& Map = D3D12::RendererSharedStates::Instance().ExternalToRegisteredTextures;
		auto  Itr = Map.find(ExternalTexture);
		if (Itr == Map.end())
		{
			Dest.reset();
		}
		else
		{
			Dest = Itr->second;
		}
	};
	FindAndSet(InSceneDepthTexture, Params.InSceneDepthTexture);
	FindAndSet(InOutColorTexture, Params.InOutColorTexture);
	assert(InOutColorTexture);
	if (InSceneDepthTexture)
	{
		assert(InSceneDepthTexture->TextureDesc.Width == InOutColorTexture->TextureDesc.Width
			&& InSceneDepthTexture->TextureDesc.Height == InOutColorTexture->TextureDesc.Height);
	}

	RenderResolution.x = InOutColorTexture->TextureDesc.Width;
	RenderResolution.y = InOutColorTexture->TextureDesc.Height;
}

void DepthBoxVDB::VolRenderer::Renderer::Unregister()
{
	switch (RHIType)
	{
		case DepthBoxVDB::VolRenderer::ERHIType::D3D12:
			for (auto ExternalTexture : { InSceneDepthTexture.get(), InOutColorTexture.get() })
				D3D12::RendererSharedStates::Instance().Unregister(ExternalTexture);
			break;
		default:
			assert(false);
	}
}

void DepthBoxVDB::VolRenderer::Renderer::SetTransferFunction(
	const TransferFunctionParameters& Params)
{
	auto Create = [&](const float* Data, const glm::uvec3& Dim) {
		auto Arr = std::make_shared<CUDA::Array>(reinterpret_cast<const float4*>(Data), Dim);

		cudaTextureDesc TexDesc{};
		TexDesc.normalizedCoords = 1;
		TexDesc.filterMode = cudaFilterModeLinear;
		TexDesc.addressMode[0] = TexDesc.addressMode[1] = TexDesc.addressMode[2] =
			cudaAddressModeBorder;
		TexDesc.readMode = cudaReadModeElementType;
		return std::make_unique<CUDA::Texture>(Arr, TexDesc);
	};
	TransferFunctionTexture =
		Create(Params.TransferFunctionData, glm::uvec3(Params.Resolution, 1, 1));
	TransferFunctionTexturePreIntegrated = Create(Params.TransferFunctionDataPreIntegrated,
		glm::uvec3(Params.Resolution, Params.Resolution, 1));
}
