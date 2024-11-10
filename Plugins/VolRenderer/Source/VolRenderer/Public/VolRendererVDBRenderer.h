#pragma once

#include "CoreMinimal.h"

#include <atomic>

#include "VolRendererUtil.h"
#include "VolDataVDB.h"

#include "DepthBoxVDB/VolRenderer.h"

#include "VolRendererVDBRenderer.generated.h"

namespace DepthBoxVDB
{
	namespace VolRenderer
	{
		inline ERHIType CastFromRHIInterfaceType(ERHIInterfaceType RHIInterfaceType)
		{
			switch (RHIInterfaceType)
			{
				case ERHIInterfaceType::D3D12:
					return ERHIType::D3D12;
			}

			return ERHIType::None;
		}

	} // namespace VolRenderer
} // namespace DepthBoxVDB

UENUM()
enum class EVolRendererRenderTarget : uint8
{
	Scene = 0 UMETA(DisplayName = "Volume Scenen"),
	AABB0 UMETA(DisplayName = "AABB of Level 0")
};

USTRUCT()
struct FVolRendererVDBRendererParameters
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere)
	EVolRendererRenderTarget RenderTarget = EVolRendererRenderTarget::Scene;
	UPROPERTY(EditAnywhere)
	bool bUseDepthBox = false;
	UPROPERTY(EditAnywhere)
	bool bUsePreIntegratedTF = true;
	UPROPERTY(EditAnywhere)
	int32 RenderResolutionY = 640;
	UPROPERTY(VisibleAnywhere)
	int32 RenderResolutionX;
	UPROPERTY(EditAnywhere)
	int32 MaxStepNum = 3000;
	UPROPERTY(EditAnywhere)
	float Step = .333f;
	UPROPERTY(EditAnywhere)
	float MaxStepDist = 1000.f;
	UPROPERTY(EditAnywhere)
	float MaxAlpha = .95f;

	float AspectRatioWOnHCached = 1.f;

	TOptional<FString> InitializeAndCheck(float AspectRatioWOnH);

	operator DepthBoxVDB::VolRenderer::VDBRendererParameters();
};

class VOLRENDERER_API FVolRendererVDBRenderer : public TSharedFromThis<FVolRendererVDBRenderer>
{
public:
	FVolRendererVDBRenderer();
	~FVolRendererVDBRenderer();

	void Register();
	void Unregister();
	void SetVDBBuilder(std::shared_ptr<DepthBoxVDB::VolData::IVDBBuilder> InVDBBuilder);
	void SetTransferFunction(
		const TArray<float>& InTransferFunctionData, const TArray<float>& InTransferFunctionDataPreIntegrated);
	void SetParameters(const FVolRendererVDBRendererParameters& Params);
	void Render_RenderThread(FPostOpaqueRenderParameters& Params);

private:
	FTextureRHIRef DepthTexture;
	FTextureRHIRef VolumeColorTexture;

	FDelegateHandle OnPostOpaqueRender;

	TArray<float> TransferFunctionData;
	TArray<float> TransferFunctionDataPreIntegrated;

	std::shared_ptr<DepthBoxVDB::VolData::IVDBBuilder>		VDBBuilder;
	std::unique_ptr<DepthBoxVDB::VolRenderer::IVDBRenderer> VDBRenderer;

	FVolRendererVDBRendererParameters VDBRendererParams;
};
