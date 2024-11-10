#include "VolRendererVDBActor.h"

AVolRendererVDBActor::AVolRendererVDBActor(const FObjectInitializer&)
{
	VDBComponent = CreateDefaultSubobject<UVolDataVDBComponent>(TEXT("VDB"));

	VDBRenderer = MakeShared<FVolRendererVDBRenderer>();

	VDBComponent->OnTransferFunctionChanged.AddLambda([this](UVolDataVDBComponent* VDBComponent) {
		auto CPUData = VDBComponent->GetCPUData();
		VDBRenderer->SetTransferFunction(CPUData->TransferFunctionData, CPUData->TransferFunctionDataPreIntegrated);
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

	FViewport* Viewport = getViewport();
	setupRenderer(Viewport);
	if (Viewport)
	{
		Viewport->ViewportResizedEvent.AddUObject(this, &AVolRendererVDBActor::setupRenderer);
	}
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
	if (PropertyChangedEvent.GetMemberPropertyName()
		== GET_MEMBER_NAME_CHECKED(AVolRendererVDBActor, VDBRendererParams))
	{
		setupRenderer(getViewport());
	}
}
#endif

FViewport* AVolRendererVDBActor::getViewport()
{
	if (auto* ViewportClient = GetWorld()->GetGameViewport(); ViewportClient)
	{
		return ViewportClient->Viewport;
	}
#if WITH_EDITOR
	else if (auto* Viewport = GEditor->GetActiveViewport(); Viewport)
	{
		return Viewport;
	}
#endif

	return nullptr;
}

void AVolRendererVDBActor::setupRenderer(FViewport* Viewport, uint32)
{
	{
		auto ErrMsgOpt = Viewport ? VDBRendererParams.InitializeAndCheck(Viewport->GetDesiredAspectRatio())
								  : VDBRendererParams.InitializeAndCheck(1.f);
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
	if (ViewportResized.IsValid())
	{
		getViewport()->ViewportResizedEvent.Remove(ViewportResized);
		ViewportResized.Reset();
	}

	clearRenderer();
}
