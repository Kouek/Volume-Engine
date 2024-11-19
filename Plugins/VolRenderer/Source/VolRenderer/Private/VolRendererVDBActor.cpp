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
			VDBRendererParamsCS.Lock();

			VDBRendererParams.Transform = SceneComponent->GetRelativeTransform();
			VDBRendererParams.InvVoxelSpaces = FVector::One() / VDBRendererParams.Transform.GetScale3D();
			auto ErrMsgOpt = VDBRendererParams.InitializeAndCheck();

			VDBRendererParamsCS.Unlock();
			if (ErrMsgOpt.IsSet())
			{
				UE_LOG(LogVolRenderer, Error, TEXT("%s"), *ErrMsgOpt.GetValue());
				return;
			}

			VDBRenderer->SetParameters(VDBRendererParams);
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

	VDBRenderer->SetVDBBuilder(VDBComponent->GetVDBBuilder());
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
