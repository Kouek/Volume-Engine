#pragma once

#include "GameFramework/Actor.h"

#include "VolDataVDB.h"
#include "VolRendererVDBRenderer.h"

#include "VolRendererVDBActor.generated.h"

UCLASS()
class VOLRENDERER_API AVolRendererVDBActor : public AActor
{
	GENERATED_BODY()

public:
	UPROPERTY(EditAnywhere)
	FVolRendererVDBRendererParameters VDBRendererParams;
	UPROPERTY(EditAnywhere)
	TObjectPtr<UVolDataVDBComponent> VDBComponent;

	AVolRendererVDBActor(const FObjectInitializer&);
	~AVolRendererVDBActor();

	void PostLoad() override;
	void Destroyed() override;
	void BeginPlay() override;

#if WITH_EDITOR
	void PostEditChangeProperty(struct FPropertyChangedEvent& PropertyChangedEvent) override;
#endif

private:
	FViewport* getViewport();
	void	   setupRenderer(FViewport* Viewport = nullptr, uint32 = 0);
	void	   clearRenderer();
	void	   clearResource();

private:
	FDelegateHandle						ViewportResized;
	TSharedPtr<FVolRendererVDBRenderer> VDBRenderer;
};
