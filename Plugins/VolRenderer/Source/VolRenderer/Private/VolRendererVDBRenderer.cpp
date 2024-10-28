#include "VolRendererVDBRenderer.h"

#include "EngineModule.h"
#include "RenderGraphBuilder.h"
#include "RenderGraphUtils.h"
#include "ShaderParameterStruct.h"

#include "Runtime/Renderer/Private/SceneRendering.h"

class VOLRENDERER_API FDepthDownsamplingCS : public FGlobalShader
{
public:
	static constexpr int32 ThreadPerGroup[2] = { 16, 16 };

	DECLARE_GLOBAL_SHADER(FDepthDownsamplingCS);
	SHADER_USE_PARAMETER_STRUCT(FDepthDownsamplingCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, VOLRENDERER_API)
	SHADER_PARAMETER(FIntPoint, InDepthSize)
	SHADER_PARAMETER(FIntPoint, OutDepthSize)
	SHADER_PARAMETER(FVector4f, InvDeviceZToWorldZTransform)
	SHADER_PARAMETER_RDG_TEXTURE_SRV(Texture2D, InDepthTexture)
	SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D, OutDepthTexture)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Params) { return true; }
	static void ModifyCompilationEnvironment(
		const FGlobalShaderPermutationParameters& Params, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Params, OutEnvironment);

		OutEnvironment.SetDefine(TEXT("THREAD_PER_GROUP_X"), ThreadPerGroup[0]);
		OutEnvironment.SetDefine(TEXT("THREAD_PER_GROUP_Y"), ThreadPerGroup[1]);
		OutEnvironment.SetDefine(TEXT("THREAD_PER_GROUP_Z"), 1);
	}
};

IMPLEMENT_GLOBAL_SHADER(FDepthDownsamplingCS, "/VolRenderer/DepthDownsampling.usf", "Main", SF_Compute);

BEGIN_SHADER_PARAMETER_STRUCT(FBarrierShaderParameters, VOLRENDERER_API)
SHADER_PARAMETER_RDG_TEXTURE(Texture2D, InDepthTexture)
SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D, OutVolumeColorTexture)
END_SHADER_PARAMETER_STRUCT()

class VOLRENDERER_API FCompositionCS : public FGlobalShader
{
public:
	static constexpr int32 ThreadPerGroup[2] = { 16, 16 };

	DECLARE_GLOBAL_SHADER(FCompositionCS);
	SHADER_USE_PARAMETER_STRUCT(FCompositionCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, VOLRENDERER_API)
	SHADER_PARAMETER(FIntPoint, RenderResolution)
	SHADER_PARAMETER_SAMPLER(SamplerState, ColorSamplerState)
	SHADER_PARAMETER_RDG_TEXTURE(Texture2D, InColorTexture)
	SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D, InOutColorTexture)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Params) { return true; }
	static void ModifyCompilationEnvironment(
		const FGlobalShaderPermutationParameters& Params, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Params, OutEnvironment);

		OutEnvironment.SetDefine(TEXT("THREAD_PER_GROUP_X"), ThreadPerGroup[0]);
		OutEnvironment.SetDefine(TEXT("THREAD_PER_GROUP_Y"), ThreadPerGroup[1]);
		OutEnvironment.SetDefine(TEXT("THREAD_PER_GROUP_Z"), 1);
	}
};

IMPLEMENT_GLOBAL_SHADER(FCompositionCS, "/VolRenderer/Composition.usf", "Main", SF_Compute);

FVolRendererVDBRenderer::FVolRendererVDBRenderer()
{
	VDBRenderer = DepthBoxVDB::VolRenderer::IVDBRenderer::Create(
		DepthBoxVDB::VolRenderer::CastFromRHIInterfaceType(GDynamicRHI->GetInterfaceType()));
}

FVolRendererVDBRenderer::~FVolRendererVDBRenderer()
{
	unregister();
}

void FVolRendererVDBRenderer::Register()
{
	unregister();

	ENQUEUE_RENDER_COMMAND(UnregisterRenderScreenRenderer)
	([Renderer = SharedThis(this)](FRHICommandListImmediate& RHICmdList) {
		GetRendererModule().RegisterPostOpaqueRenderDelegate(
			FPostOpaqueRenderDelegate::CreateRaw(&Renderer.Get(), &FVolRendererVDBRenderer::Render_RenderThread));
	});
}

void FVolRendererVDBRenderer::SetParameters(const FVolRendererVDBRendererParameters& Params)
{
	VDBRendererParams = Params;
}

void FVolRendererVDBRenderer::Render_RenderThread(FPostOpaqueRenderParameters& Params)
{
	if (!VDBRenderer)
		return;

	auto* GraphBuilder = Params.GraphBuilder;
	auto& RHICmdList = GraphBuilder->RHICmdList;

	void*	  InDepthTextureNative = nullptr;
	void*	  OutVolumeColorTextureNative = nullptr;
	FIntPoint VolumeRenderResolution =
		FIntPoint(VDBRendererParams.RenderResolutionX, VDBRendererParams.RenderResolutionY);
	{
		constexpr ETextureCreateFlags NeededTextureCreateFlags = ETextureCreateFlags::Shared | ETextureCreateFlags::UAV;

		bool bNeedRegister = false;
		if (!DepthTexture.IsValid() || DepthTexture->GetDesc().Extent.X != VDBRendererParams.RenderResolutionX
			|| DepthTexture->GetDesc().Extent.Y != VDBRendererParams.RenderResolutionY)
		{
			VDBRenderer->Unregister();

			auto Desc =
				FRHITextureCreateDesc::Create2D(UE_SOURCE_LOCATION, VolumeRenderResolution, EPixelFormat::PF_R32_FLOAT);
			Desc.AddFlags(NeededTextureCreateFlags);
			DepthTexture = RHICmdList.CreateTexture(Desc);
			bNeedRegister = true;
		}
		if (!VolumeColorTexture.IsValid()
			|| VolumeColorTexture->GetDesc().Extent.X != VDBRendererParams.RenderResolutionX
			|| VolumeColorTexture->GetDesc().Extent.Y != VDBRendererParams.RenderResolutionY)
		{
			VDBRenderer->Unregister();

			auto Desc =
				FRHITextureCreateDesc::Create2D(UE_SOURCE_LOCATION, VolumeRenderResolution, EPixelFormat::PF_R8G8B8A8);
			Desc.AddFlags(NeededTextureCreateFlags);
			VolumeColorTexture = RHICmdList.CreateTexture(Desc);
			bNeedRegister = true;
		}

		InDepthTextureNative = DepthTexture->GetNativeResource();
		OutVolumeColorTextureNative = VolumeColorTexture->GetNativeResource();
		if (bNeedRegister)
		{
			VDBRenderer->Register({ .Device = GDynamicRHI->RHIGetNativeDevice(),
				.InDepthTexture = InDepthTextureNative,
				.OutColorTexture = OutVolumeColorTextureNative });
		}
	}

	auto DepthTextureRDG = RegisterExternalTexture(*GraphBuilder, DepthTexture, UE_SOURCE_LOCATION);
	{
		auto DepthTextureRDGUAV = GraphBuilder->CreateUAV(FRDGTextureUAVDesc(DepthTextureRDG));
		auto ShaderParams = GraphBuilder->AllocParameters<FDepthDownsamplingCS::FParameters>();
		ShaderParams->InDepthSize = Params.ViewportRect.Size();
		ShaderParams->OutDepthSize = VolumeRenderResolution;
		ShaderParams->InvDeviceZToWorldZTransform = Params.View->InvDeviceZToWorldZTransform;
		ShaderParams->InDepthTexture = GraphBuilder->CreateSRV(FRDGTextureSRVDesc::Create(Params.DepthTexture));
		ShaderParams->OutDepthTexture = DepthTextureRDGUAV;

		TShaderMapRef<FDepthDownsamplingCS> Shader(GetGlobalShaderMap(GMaxRHIFeatureLevel));
		FComputeShaderUtils::AddPass(*GraphBuilder, RDG_EVENT_NAME("Depth Down Sampling"),
			ERDGPassFlags::AsyncCompute | ERDGPassFlags::NeverCull, Shader, ShaderParams,
			FIntVector(FMath::DivideAndRoundUp(VolumeRenderResolution.X, FDepthDownsamplingCS::ThreadPerGroup[0]),
				FMath::DivideAndRoundUp(VolumeRenderResolution.Y, FDepthDownsamplingCS::ThreadPerGroup[1]), 1));
	}

	auto VolumeColorTextureRDG = RegisterExternalTexture(*GraphBuilder, VolumeColorTexture, UE_SOURCE_LOCATION);
	{
		auto VolumeColorTextureRDGUAV = GraphBuilder->CreateUAV(FRDGTextureUAVDesc(VolumeColorTextureRDG));
		auto ShaderParams = GraphBuilder->AllocParameters<FBarrierShaderParameters>();
		ShaderParams->InDepthTexture = DepthTextureRDG;
		ShaderParams->OutVolumeColorTexture = VolumeColorTextureRDGUAV;

		auto ShaderParametersMetadata = FBarrierShaderParameters::FTypeInfo::GetStructMetadata();
		GraphBuilder->AddPass(RDG_EVENT_NAME("Volume Rendering"), ShaderParametersMetadata, ShaderParams,
			ERDGPassFlags::AsyncCompute | ERDGPassFlags::NeverCull,
			[DepthTexture = DepthTexture, VolumeColorTexture = VolumeColorTexture, VDBRenderer = VDBRenderer.get()](
				FRHICommandListImmediate& RHICmdList) { VDBRenderer->Render({ .bUseDepthBox = true }); });
	}

	{
		auto ShaderParams = GraphBuilder->AllocParameters<FCompositionCS::FParameters>();
		ShaderParams->RenderResolution = Params.ViewportRect.Size();
		ShaderParams->ColorSamplerState = TStaticSamplerState<SF_Bilinear>::GetRHI();
		ShaderParams->InColorTexture = VolumeColorTextureRDG;
		ShaderParams->InOutColorTexture = GraphBuilder->CreateUAV(FRDGTextureUAVDesc(Params.ColorTexture));

		TShaderMapRef<FCompositionCS> Shader(GetGlobalShaderMap(GMaxRHIFeatureLevel));
		FComputeShaderUtils::AddPass(*GraphBuilder, RDG_EVENT_NAME("Composition"),
			ERDGPassFlags::AsyncCompute | ERDGPassFlags::NeverCull, Shader, ShaderParams,
			FIntVector(
				FMath::DivideAndRoundUp(ShaderParams->RenderResolution.X, FDepthDownsamplingCS::ThreadPerGroup[0]),
				FMath::DivideAndRoundUp(ShaderParams->RenderResolution.Y, FDepthDownsamplingCS::ThreadPerGroup[1]), 1));
	}
}

void FVolRendererVDBRenderer::unregister()
{
	if (!OnPostOpaqueRender.IsValid())
		return;

	ENQUEUE_RENDER_COMMAND(UnregisterRenderScreenRenderer)
	([Renderer = SharedThis(this)](FRHICommandListImmediate& RHICmdList) {
		GetRendererModule().RemovePostOpaqueRenderDelegate(Renderer->OnPostOpaqueRender);
		Renderer->OnPostOpaqueRender.Reset();
	});
}
