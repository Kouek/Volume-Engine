#pragma once

#include "GameFramework/Actor.h"

#include "VolDataVDB.h"
#include "VolDeformTetrahedralActor.h"
#include "VolRendererVDBRenderer.h"

#include "VolRendererVDBActor.generated.h"

UCLASS()
class VOLRENDERER_API AVolRendererVDBActor : public AActor
{
	GENERATED_BODY()

public:
	UPROPERTY(EditAnywhere, Category = "VolRenderer", DisplayName = "VDB Renderer Parameters")
	FVolRendererVDBRendererParameters VDBRendererParams;
	UPROPERTY(VisibleAnywhere, Category = "VolRenderer", DisplayName = "VDB")
	TObjectPtr<UVolDataVDBComponent> VDBComponent;

	UPROPERTY(EditAnywhere, Transient, Category = "VolDeform")
	int32 CurrentFrameIndex = 0;
	UPROPERTY(EditAnywhere, Category = "VolDeform")
	TObjectPtr<AVolDeformTetrahedralActor> TetrahedralActor;

	AVolRendererVDBActor(const FObjectInitializer&);
	~AVolRendererVDBActor();

	void PostLoad() override;
	void Destroyed() override;
	void BeginPlay() override;

#if WITH_EDITOR
	void PostEditChangeProperty(struct FPropertyChangedEvent& PropertyChangedEvent) override;
#endif

private:
	void updateVoxelSpaces();
	void updateVisibleBox();

	void setupRenderer();
	void clearRenderer();
	void clearResource();

private:
	TSharedPtr<FVolRendererVDBRenderer> VDBRenderer;

	FCriticalSection VDBRendererParamsCS;
};
