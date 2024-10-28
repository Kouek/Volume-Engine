#include "VolRendererVDBActor.h"

AVolRendererVDBActor::AVolRendererVDBActor(const FObjectInitializer&)
{
	VDBComponent = CreateDefaultSubobject<UVolDataVDBComponent>(TEXT("VDB"));
}

void AVolRendererVDBActor::PostLoad()
{
	Super::PostLoad();

	setupRenderer();
}

void AVolRendererVDBActor::Destroyed()
{
	VDBRenderer.Reset();

	Super::Destroyed();
}

#if WITH_EDITOR
void AVolRendererVDBActor::PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent)
{
	if (PropertyChangedEvent.GetMemberPropertyName()
		== GET_MEMBER_NAME_CHECKED(AVolRendererVDBActor, VDBRendererParams))
	{
		setupRenderer();
	}
}
#endif

void AVolRendererVDBActor::setupRenderer()
{
	if (!VDBRenderer.IsValid())
	{
		VDBRenderer = MakeShared<FVolRendererVDBRenderer>();
		VDBRenderer->Register();
	}

	if (auto* ViewportClient = GetWorld()->GetGameViewport(); ViewportClient)
	{
		VDBRendererParams.InitializeAndCheck(ViewportClient->Viewport->GetDesiredAspectRatio());
	}
	else
	{
		VDBRendererParams.InitializeAndCheck(1.f);
	}
	VDBRenderer->SetParameters(VDBRendererParams);
}
