#include "VolRenderer.h"

std::unique_ptr<DepthBoxVDB::VolRenderer::IVDBRenderer>
DepthBoxVDB::VolRenderer::IVDBRenderer::Create(ERHIType RHIType)
{
	if (RHIType == ERHIType::D3D12)
	{
		return std::make_unique<VDBRendererImpl>(RHIType);
	}
	return {};
}

DepthBoxVDB::VolRenderer::VDBRendererImpl::VDBRendererImpl(ERHIType RHIType) : RHIType(RHIType)
{
	int DeviceNum = 0;
	DEPTHBOXVDB_CHECK(cudaGetDeviceCount(&DeviceNum));
	assert(DeviceNum > 0);

	cudaDeviceProp Prop;
	DEPTHBOXVDB_CHECK(cudaGetDeviceProperties(&Prop, 0));
	D3D12NodeMask = Prop.luidDeviceNodeMask;

	DEPTHBOXVDB_CHECK(cudaStreamCreate(&Stream));
}

void DepthBoxVDB::VolRenderer::VDBRendererImpl::Register(const RendererParameters& Params)
{
	ID3D12Device*	Device = reinterpret_cast<ID3D12Device*>(Params.Device);
	ID3D12Resource* InDepthTextureNative = reinterpret_cast<ID3D12Resource*>(Params.InDepthTexture);
	ID3D12Resource* OutColorTextureNative =
		reinterpret_cast<ID3D12Resource*>(Params.OutColorTexture);

	InDepthTexture =
		std::make_unique<D3D12TextureInteropCUDA>(D3D12NodeMask, Device, InDepthTextureNative);
	OutColorTexture =
		std::make_unique<D3D12TextureInteropCUDA>(D3D12NodeMask, Device, OutColorTextureNative);
	RenderResolution.x = InDepthTexture->TextureDesc.Width;
	RenderResolution.y = InDepthTexture->TextureDesc.Height;
	assert(RenderResolution.x == OutColorTexture->TextureDesc.Width
		&& RenderResolution.y == OutColorTexture->TextureDesc.Height);
}

void DepthBoxVDB::VolRenderer::VDBRendererImpl::Unregister()
{
	InDepthTexture.reset();
	OutColorTexture.reset();
}

void DepthBoxVDB::VolRenderer::VDBRendererImpl::Render(const RenderParameters& Params)
{
	if (!InDepthTexture || !OutColorTexture)
		return;

	dim3	   ThreadPerBlock(16, 16, 1);
	dim3	   BlockPerGrid((RenderResolution.x + ThreadPerBlock.x - 1) / ThreadPerBlock.x,
			  (RenderResolution.y + ThreadPerBlock.y - 1) / ThreadPerBlock.y);
	static int Time = 0;
	ParallelFor(
		BlockPerGrid, ThreadPerBlock,
		[Time = Time, RenderResolution = RenderResolution,
			InDepthSurface = InDepthTexture->SurfaceObject,
			OutColorSurface =
				OutColorTexture->SurfaceObject] __device__(const glm::uvec3& DispatchThreadID) {
			if (DispatchThreadID.x >= RenderResolution.x
				|| DispatchThreadID.y >= RenderResolution.y)
				return;

			float Depth = surf2Dread<float>(
				InDepthSurface, sizeof(float) * DispatchThreadID.x, DispatchThreadID.y);
			Depth = 0.f;

			glm::vec4 Color(static_cast<float>((DispatchThreadID.x + Time) % RenderResolution.x)
					/ (RenderResolution.x - 1),
				static_cast<float>((DispatchThreadID.y + 2 * Time) % RenderResolution.y)
					/ (RenderResolution.y - 1),
				Depth / 1024.f, .5f);
			Color = glm::clamp(Color * 255.f, 0.f, 255.f);
			uchar4 ColorUCh4{ Color.r, Color.g, Color.b, Color.a };

			surf2Dwrite(ColorUCh4, OutColorSurface, sizeof(uchar4) * DispatchThreadID.x,
				DispatchThreadID.y);
		},
		Stream);
	++Time;
	if (Time == RenderResolution.x)
		Time = 0;

	DEPTHBOXVDB_CHECK(cudaStreamSynchronize(Stream));
}
