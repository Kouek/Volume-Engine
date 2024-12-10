#include "VolRendererVDBActor.h"

AVolRendererVDBActor::AVolRendererVDBActor(const FObjectInitializer&)
{
	VDBComponent = CreateDefaultSubobject<UVolDataVDBComponent>(TEXT("VDB"));
	SetRootComponent(VDBComponent);
	VDBComponent->TransferFunctionChanged.AddLambda([this](UVolDataVDBComponent* VDBComponent) {
		auto CPUData = VDBComponent->GetCPUData();
		VDBRenderer->SetTransferFunction(CPUData->TransferFunctionData, CPUData->TransferFunctionDataPreIntegrated);
	});
	VDBComponent->TransformUpdated.AddLambda(
		[this](USceneComponent* SceneComponent, EUpdateTransformFlags, ETeleportType) {
			updateVoxelSpaces();
			updateVisibleBox();
			setupRenderer();
		});

	VDBRenderer = MakeShared<FVolRendererVDBRenderer>();
	VDBRenderer->RenderSizeChanged_RenderThread.AddLambda([this](FIntPoint ActualRenderResolution) {
		VDBRendererParamsCS.Lock();

		VDBRendererParams.RenderResolution = ActualRenderResolution;

		VDBRendererParamsCS.Unlock();
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

void AVolRendererVDBActor::Destroyed()
{
	clearResource();

	Super::Destroyed();
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

	if (PropertyChangedEvent.GetMemberPropertyName()
		== GET_MEMBER_NAME_CHECKED(AVolRendererVDBActor, CurrentFrameIndex))
	{
		VolRenderer::FStdOutputLinker Linker;
		VDBComponent->GetVDB()->SwitchToFrame(CurrentFrameIndex);
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
		VDBRendererParamsCS.Lock();
		auto ErrMsgOpt = VDBRendererParams.InitializeAndCheck();
		VDBRendererParamsCS.Unlock();
		if (ErrMsgOpt.IsSet())
		{
			UE_LOG(LogVolRenderer, Error, TEXT("%s"), *ErrMsgOpt.GetValue());
			return;
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
	clearRenderer();
}

void AVolRendererVDBActor::updateVoxelSpaces()
{
	VDBRendererParamsCS.Lock();

	VDBRendererParams.Transform = VDBComponent->GetRelativeTransform();
	VDBRendererParams.InvVoxelSpaces = FVector::One() / VDBRendererParams.Transform.GetScale3D();

	VDBRendererParamsCS.Unlock();
}

void AVolRendererVDBActor::updateVisibleBox()
{
	if (!TetrahedralActor)
	{
		VDBRendererParamsCS.Lock();
		VDBRendererParams.ResetVisibleBox();
		VDBRendererParamsCS.Unlock();

		return;
	}

	VDBRendererParamsCS.Lock();

	const FTransform& TATr = TetrahedralActor->GetTransform();
	const FTransform& VDBTr = GetTransform();
	VDBRendererParams.VisibleBoxMinPositionToLocal = TATr.GetLocation() - VDBTr.GetLocation();
	VDBRendererParams.VisibleBoxMaxPositionToLocal =
		VDBRendererParams.VisibleBoxMinPositionToLocal + TetrahedralActor->TetrahedralMeshParams.Extent;

	VDBRendererParamsCS.Unlock();
}
