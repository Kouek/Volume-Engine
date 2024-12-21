#include "VolRendererVDBActor.h"

AVolRendererVDBActor::AVolRendererVDBActor(const FObjectInitializer&)
{
	VDBComponent = CreateDefaultSubobject<UVolDataVDBComponent>(TEXT("VDB"));
	SetRootComponent(VDBComponent);
	OnTransferFunctionChanged = VDBComponent->TransferFunctionChanged.AddLambda([this]() {
		auto CPUData = VDBComponent->GetCPUData();
		VDBRenderer->SetTransferFunction(CPUData->TransferFunctionData, CPUData->TransferFunctionDataPreIntegrated);
	});
	OnTransformUpdated = VDBComponent->TransformUpdated.AddLambda(
		[this](USceneComponent* SceneComponent, EUpdateTransformFlags, ETeleportType) {
			updateVoxelSpaces();
			updateVisibleBox();
			setupRenderer();
		});

	VDBRenderer = MakeShared<FVolRendererVDBRenderer>();
	OnRenderSizeChanged_RenderThread =
		VDBRenderer->RenderSizeChanged_RenderThread.AddLambda([this](FIntPoint ActualRenderResolution) {
			FScopeLock ParamsSL(&ParamsCS);

			VDBRendererParams.RenderResolution = ActualRenderResolution;
		});
}

AVolRendererVDBActor::~AVolRendererVDBActor()
{
	clearResource();
}

void AVolRendererVDBActor::PostLoad()
{
	Super::PostLoad();

	VDBRenderer->Register();

	if (TetrahedralActor)
	{
		TetrahedralActor->BoundingBoxChanged.AddLambda([this](AVolDeformTetrahedralActor*) {
			updateVisibleBox();
			setupRenderer();
		});
	}

	updateVoxelSpaces();
	updateVisibleBox();
	setupRenderer();
}

void AVolRendererVDBActor::BeginPlay() {}

#if WITH_EDITOR
void AVolRendererVDBActor::PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent)
{
	Super::PostEditChangeProperty(PropertyChangedEvent);

	if (PropertyChangedEvent.GetMemberPropertyName()
		== GET_MEMBER_NAME_CHECKED(AVolRendererVDBActor, VDBRendererParams))
	{
		setupRenderer();
	}

	if (PropertyChangedEvent.GetMemberPropertyName() == GET_MEMBER_NAME_CHECKED(AVolRendererVDBActor, TetrahedralActor))
	{
		if (TetrahedralActor)
		{
			TetrahedralActor->BoundingBoxChanged.AddLambda([this](AVolDeformTetrahedralActor*) {
				updateVisibleBox();
				setupRenderer();
			});
		}

		updateVisibleBox();
		setupRenderer();
	}
}
#endif

void AVolRendererVDBActor::setupRenderer()
{
	{
		{
			FScopeLock ParamsSL(&ParamsCS);

			auto ErrMsgOpt = VDBRendererParams.InitializeAndCheck();
			if (ErrMsgOpt.IsSet())
			{
				UE_LOG(LogVolRenderer, Error, TEXT("%s"), *ErrMsgOpt.GetValue());
				return;
			}
		}

		VDBRenderer->SetParameters(VDBRendererParams);
	}

	VDBRenderer->SetVDB(VDBComponent->GetVDB());
}

void AVolRendererVDBActor::clearRenderer()
{
	if (VDBRenderer)
	{
		VDBRenderer->Unregister();
		VDBRenderer.Reset();
	}
}

void AVolRendererVDBActor::clearResource()
{
	OnVDBChanged.Reset();
	OnTransferFunctionChanged.Reset();
	OnTransformUpdated.Reset();

	OnRenderSizeChanged_RenderThread.Reset();
	OnFrameIndexChanged_RenderThread.Reset();

	clearRenderer();
}

void AVolRendererVDBActor::updateVoxelSpaces()
{
	FScopeLock ParamsSL(&ParamsCS);

	VDBRendererParams.Transform = VDBComponent->GetRelativeTransform();
	VDBRendererParams.VoxelSpaces = VDBRendererParams.Transform.GetScale3D();
	VDBRendererParams.InvVoxelSpaces = FVector::One() / VDBRendererParams.VoxelSpaces;
}

void AVolRendererVDBActor::updateVisibleBox()
{
	FScopeLock ParamsSL(&ParamsCS);

	if (!TetrahedralActor)
	{
		VDBRendererParams.ResetVisibleBox();

		return;
	}

	const FTransform& TATr = TetrahedralActor->GetTransform();
	const FTransform& VDBTr = GetTransform();
	VDBRendererParams.VisibleBoxMinPositionToLocal = TATr.GetLocation() - VDBTr.GetLocation();
	VDBRendererParams.VisibleBoxMaxPositionToLocal =
		VDBRendererParams.VisibleBoxMinPositionToLocal + TetrahedralActor->TetrahedralMeshParams.Extent;
}
