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

			check(false);
			return ERHIType::D3D12;
		}

	} // namespace VolRenderer
} // namespace DepthBoxVDB

UENUM()
enum class EVolRendererRenderTarget : uint8
{
	Scene = 0 UMETA(DisplayName = "Volume Scenen"),
	AABB0	   UMETA(DisplayName = "AABB of Level 0"),
	AABB1	   UMETA(DisplayName = "AABB of Level 1"),
	AABB2	   UMETA(DisplayName = "AABB of Level 2"),
	DepthBox   UMETA(DisplayName = "Depth Box"),
	PixelDepth UMETA(DisplayName = "Input Pixel Depth")
};

USTRUCT()
struct FVolRendererVDBRendererParameters
{
	GENERATED_BODY()

	static constexpr int32 kMaxResolutionLOD = 4;

	UPROPERTY(VisibleAnywhere)
	FIntPoint RenderResolution = { 0, 0 };
	UPROPERTY(EditAnywhere)
	EVolRendererRenderTarget RenderTarget = EVolRendererRenderTarget::Scene;
	UPROPERTY(EditAnywhere)
	bool bUseDepthBox = false;
	UPROPERTY(EditAnywhere)
	bool bUsePreIntegratedTF = true;
	UPROPERTY(EditAnywhere)
	bool bUseDepthOcclusion = true;
	UPROPERTY(EditAnywhere)
	bool bClipByVisibleBox = false;
	bool bRenderDeformVolume = false;
	UPROPERTY(EditAnywhere)
	int32 RenderResolutionLOD = 1;
	UPROPERTY(EditAnywhere)
	int32 MaxStepNum = 3000;
	UPROPERTY(EditAnywhere)
	float Step = .333f;
	UPROPERTY(EditAnywhere)
	float MaxStepDist = 3000.f;
	UPROPERTY(EditAnywhere)
	float MaxAlpha = .95f;
	UPROPERTY(VisibleAnywhere)
	FVector InvVoxelSpaces = { 1., 1., 1. };
	UPROPERTY(VisibleAnywhere)
	FVector VisibleBoxMinPositionToLocal = FVector::OneVector;
	UPROPERTY(VisibleAnywhere)
	FVector VisibleBoxMaxPositionToLocal = FVector::ZeroVector;
	FVector DeformVolumeOrigin = FVector::ZeroVector;

	FTransform Transform;

	void ResetVisibleBox()
	{
		VisibleBoxMinPositionToLocal = FVector::OneVector;
		VisibleBoxMaxPositionToLocal = FVector::ZeroVector;
	}

	TOptional<FString> InitializeAndCheck();

	operator DepthBoxVDB::VolRenderer::IVDBRenderer::RendererParameters();
};

class VOLRENDERER_API FVolRendererVDBRenderer : public TSharedFromThis<FVolRendererVDBRenderer>
{
public:
	static void AssignLeftHandedToRight(glm::vec3& Rht, const FVector& Lft);

	FVolRendererVDBRenderer();
	~FVolRendererVDBRenderer();

	void Register();
	void Unregister();
	void SetVDB(std::shared_ptr<DepthBoxVDB::VolData::IVDB> InVDB);
	void SetTransferFunction(
		const TArray<float>& InTransferFunctionData, const TArray<float>& InTransferFunctionDataPreIntegrated);
	void SetParameters(const FVolRendererVDBRendererParameters& Params);
	void Render_RenderThread(FPostOpaqueRenderParameters& Params);

	DECLARE_MULTICAST_DELEGATE_OneParam(FRenderSizeChanged, FIntPoint);

	FRenderSizeChanged RenderSizeChanged_RenderThread;

private:
	bool bCanLogErrInRender_RenderThread = true; // Avoid too much Error Logs when rendering

	FTextureRHIRef DepthTexture;
	FTextureRHIRef VolumeColorTexture;

	FDelegateHandle OnPostOpaqueRender;

	TArray<float> TransferFunctionData;
	TArray<float> TransferFunctionDataPreIntegrated;

	std::shared_ptr<DepthBoxVDB::VolData::IVDB>				VDB;
	std::unique_ptr<DepthBoxVDB::VolRenderer::IVDBRenderer> VDBRenderer;

	FVolRendererVDBRendererParameters VDBRendererParams;
};
