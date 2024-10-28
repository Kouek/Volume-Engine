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
	UVolDataVDBComponent* VDBComponent;

	AVolRendererVDBActor(const FObjectInitializer&);

	void PostLoad() override;
	void Destroyed() override;

#if WITH_EDITOR
	void PostEditChangeProperty(struct FPropertyChangedEvent& PropertyChangedEvent) override;
#endif

private:
	void setupRenderer();

private:
	TSharedPtr<FVolRendererVDBRenderer> VDBRenderer;
};
