#include "VolRendererVDBRenderer.h"

#include "EngineModule.h"
#include "RenderGraphBuilder.h"
#include "RenderGraphUtils.h"
#include "ShaderParameterStruct.h"

#include "Runtime/Renderer/Private/SceneRendering.h"

TOptional<FString> FVolRendererVDBRendererParameters::InitializeAndCheck(float AspectRatioWOnH)
{
	AspectRatioWOnHCached = AspectRatioWOnH;

#define CHECK(Member, Min, Max)                                                 \
	if (Member < Min || Member > Max)                                           \
	{                                                                           \
		return FString::Format(TEXT("Invalid " #Member " = {0}."), { Member }); \
	}

	CHECK(RenderResolutionY, 100, 10000)
	RenderResolutionX = FMath::RoundToInt(AspectRatioWOnHCached * RenderResolutionY);
	CHECK(MaxStepNum, 1, 10000)
	CHECK(Step, 0.f, std::numeric_limits<float>::max())
	CHECK(MaxStepDist, 0.f, std::numeric_limits<float>::max())
	CHECK(MaxAlpha, .1f, 1.f);

#undef CHECK

	return {};
}

FVolRendererVDBRendererParameters::operator DepthBoxVDB::VolRenderer::VDBRendererParameters()
{
	DepthBoxVDB::VolRenderer::VDBRendererParameters Ret;

#define ASSIGN(Member) Ret.Member = Member

	Ret.RenderTarget = (DepthBoxVDB::VolRenderer::ERenderTarget)(uint8)RenderTarget;
	ASSIGN(bUseDepthBox);
	ASSIGN(bUsePreIntegratedTF);
	ASSIGN(MaxStepNum);
	ASSIGN(Step);
	ASSIGN(MaxStepDist);
	ASSIGN(MaxAlpha);

#undef ASSIGN

	return Ret;
}

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
		{ .RHIType = DepthBoxVDB::VolRenderer::CastFromRHIInterfaceType(GDynamicRHI->GetInterfaceType()) });
}

FVolRendererVDBRenderer::~FVolRendererVDBRenderer() {}

void FVolRendererVDBRenderer::Register()
{
	Unregister();

	ENQUEUE_RENDER_COMMAND(UnregisterRenderScreenRenderer)
	([Renderer = SharedThis(this)](FRHICommandListImmediate& RHICmdList) {
		Renderer->OnPostOpaqueRender = GetRendererModule().RegisterPostOpaqueRenderDelegate(
			FPostOpaqueRenderDelegate::CreateRaw(&Renderer.Get(), &FVolRendererVDBRenderer::Render_RenderThread));
	});
}

void FVolRendererVDBRenderer::Unregister()
{
	if (!OnPostOpaqueRender.IsValid())
		return;

	ENQUEUE_RENDER_COMMAND(UnregisterRenderScreenRenderer)
	([Renderer = SharedThis(this)](FRHICommandListImmediate& RHICmdList) {
		GetRendererModule().RemovePostOpaqueRenderDelegate(Renderer->OnPostOpaqueRender);
		Renderer->OnPostOpaqueRender.Reset();
		Renderer->VDBRenderer->Unregister();
	});
}

void FVolRendererVDBRenderer::SetVDBBuilder(std::shared_ptr<DepthBoxVDB::VolData::IVDBBuilder> InVDBBuilder)
{
	ENQUEUE_RENDER_COMMAND(SetVDBBuilder)
	([Renderer = SharedThis(this), InVDBBuilder](
		 FRHICommandListImmediate& RHICmdList) { Renderer->VDBBuilder = InVDBBuilder; });
}

void FVolRendererVDBRenderer::SetTransferFunction(
	const TArray<float>& InTransferFunctionData, const TArray<float>& InTransferFunctionDataPreIntegrated)
{
	TransferFunctionData = InTransferFunctionData;
	TransferFunctionDataPreIntegrated = InTransferFunctionDataPreIntegrated;

	ENQUEUE_RENDER_COMMAND(SetTransferFunction)
	([Renderer = SharedThis(this)](FRHICommandListImmediate& RHICmdList) {
		if (!Renderer->VDBRenderer)
			return;

		VolRenderer::FStdOutputLinker Linker;
		Renderer->VDBRenderer->SetTransferFunction({ .TransferFunctionData = Renderer->TransferFunctionData.GetData(),
			.TransferFunctionDataPreIntegrated = Renderer->TransferFunctionDataPreIntegrated.GetData(),
			.Resolution = static_cast<uint32>(Renderer->TransferFunctionData.Num() / 4) /* float32 RGBA */ });
	});
}

void FVolRendererVDBRenderer::SetParameters(const FVolRendererVDBRendererParameters& Params)
{
	ENQUEUE_RENDER_COMMAND(SetParameters)
	([Renderer = SharedThis(this), Params](FRHICommandListImmediate& RHICmdList) {
		if (!Renderer->VDBRenderer)
			return;

		Renderer->VDBRendererParams = Params;
		VolRenderer::FStdOutputLinker Linker;
		Renderer->VDBRenderer->SetParameters(Renderer->VDBRendererParams);
	});
}

void FVolRendererVDBRenderer::Render_RenderThread(FPostOpaqueRenderParameters& Params)
{
	if (!VDBRenderer || !VDBBuilder)
		return;

	auto* GraphBuilder = Params.GraphBuilder;
	auto& RHICmdList = GraphBuilder->RHICmdList;

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

		void* InDepthTextureNative = DepthTexture->GetNativeResource();
		void* OutVolumeColorTextureNative = VolumeColorTexture->GetNativeResource();
		if (bNeedRegister)
		{
			VolRenderer::FStdOutputLinker Linker;
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

		/*
		 * | 0 1 0|   |x|   | y|
		 * | 0 0 1| * |y| = | z|
		 * |-1 0 0|   |z|L  |-x|R
		 */
		auto AssignLeftHandedToRight = [](glm::vec3& Rht, const FVector& Lft) {
			Rht.x = +Lft.Y;
			Rht.y = +Lft.Z;
			Rht.z = -Lft.X;
		};
		glm::vec3 CameraPositionToLoacl;
		{
			const FVector& CameraPos = Params.View->ViewMatrices.GetViewOrigin();
			AssignLeftHandedToRight(CameraPositionToLoacl, CameraPos);
		}

		glm::mat3 CameraRotationToLoacl;
		{
			AssignLeftHandedToRight(CameraRotationToLoacl[2], Params.View->GetViewDirection());
			CameraRotationToLoacl[2] *= -1.f;
			AssignLeftHandedToRight(CameraRotationToLoacl[0], Params.View->GetViewRight());
			AssignLeftHandedToRight(CameraRotationToLoacl[1], Params.View->GetViewUp());
		}

		const FMatrix& InvProjMatrix = Params.View->ViewMatrices.GetInvProjectionMatrix();
		glm::mat4	   InverseProjection(InvProjMatrix.M[0][0], InvProjMatrix.M[0][1], InvProjMatrix.M[0][2],
				 InvProjMatrix.M[0][3], InvProjMatrix.M[1][0], InvProjMatrix.M[1][1], InvProjMatrix.M[1][2],
				 InvProjMatrix.M[1][3], InvProjMatrix.M[2][0], InvProjMatrix.M[2][1], InvProjMatrix.M[2][2],
				 InvProjMatrix.M[2][3], InvProjMatrix.M[3][0], InvProjMatrix.M[3][1], -InvProjMatrix.M[3][2],
				 InvProjMatrix.M[3][3]);

		GraphBuilder->AddPass(RDG_EVENT_NAME("Volume Rendering"), ShaderParametersMetadata, ShaderParams,
			ERDGPassFlags::AsyncCompute | ERDGPassFlags::NeverCull,
			[InverseProjection, CameraRotationToLoacl, CameraPositionToLoacl, DepthTexture = DepthTexture,
				VolumeColorTexture = VolumeColorTexture, VDBRenderer = VDBRenderer.get(),
				VDBBuilder = VDBBuilder.get()](FRHICommandListImmediate& RHICmdList) {
				VolRenderer::FStdOutputLinker Linker;
				VDBRenderer->Render({ .InverseProjection = InverseProjection,
					.CameraRotationToLocal = CameraRotationToLoacl,
					.CameraPositionToLocal = CameraPositionToLoacl,
					.Builder = *VDBBuilder });
			});
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
