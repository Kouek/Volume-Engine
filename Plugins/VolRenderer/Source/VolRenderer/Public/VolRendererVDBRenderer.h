#pragma once

#include "CoreMinimal.h"

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

USTRUCT()
struct FVolRendererVDBRendererParameters
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere)
	int RenderResolutionY = 100;
	UPROPERTY(VisibleAnywhere)
	int RenderResolutionX;
	UPROPERTY(EditAnywhere)
	int MaxStepNum = 3000;
	UPROPERTY(EditAnywhere)
	float Step = .5f / 3000;
	UPROPERTY(EditAnywhere)
	float MaxStepDist = 1000.f;
	UPROPERTY(VisibleAnywhere)
	FMatrix VolumeToWorld;

	TOptional<FString> InitializeAndCheck(float AspectRatioWOnH)
	{
#define CHECK(Member, Min, Max)                                                 \
	if (Member < Min || Member > Max)                                           \
	{                                                                           \
		return FString::Format(TEXT("Invalid " #Member " = {0}."), { Member }); \
	}

		CHECK(RenderResolutionY, 100, 10000)
		RenderResolutionX = FMath::RoundToInt(AspectRatioWOnH * RenderResolutionY);
		CHECK(MaxStepNum, 1, 10000)
		CHECK(Step, 0.f, std::numeric_limits<float>::max())
		CHECK(MaxStepDist, 0.f, std::numeric_limits<float>::max())

#undef CHECK

		return {};
	}
};

class VOLRENDERER_API FVolRendererVDBRenderer : public TSharedFromThis<FVolRendererVDBRenderer>
{
public:
	FVolRendererVDBRenderer();
	~FVolRendererVDBRenderer();

	void Register();
	void SetParameters(const FVolRendererVDBRendererParameters& Params);
	void Render_RenderThread(FPostOpaqueRenderParameters& Params);

private:
	void unregister();

private:
	FTextureRHIRef	DepthTexture;
	FTextureRHIRef	VolumeColorTexture;
	FDelegateHandle OnPostOpaqueRender;

	std::unique_ptr<DepthBoxVDB::VolRenderer::IVDBRenderer> VDBRenderer;

	FVolRendererVDBRendererParameters VDBRendererParams;
};
