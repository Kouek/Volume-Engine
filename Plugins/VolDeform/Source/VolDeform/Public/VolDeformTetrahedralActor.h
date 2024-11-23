#pragma once

#include "GameFramework/Actor.h"
#include "Components/InstancedStaticMeshComponent.h"

#include "VolDeformUtil.h"

#include "VolDeformTetrahedralActor.generated.h"

USTRUCT()
struct FVolDeformTetrahedralMeshParameters
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere)
	FIntVector Resolution = { 5, 5, 5 };
	UPROPERTY(EditAnywhere)
	FVector Extent = { 100.f, 100.f, 100.f };
	UPROPERTY(EditAnywhere)
	float NodeScale = 1 / 25.f;

	TOptional<FString> InitializeAndCheck();
};

UCLASS()
class VOLDEFORM_API AVolDeformTetrahedralActor : public AActor
{
	GENERATED_BODY()

public:
	UPROPERTY(EditAnywhere, Category = "VolDeform", DisplayName = "Tetrahedral Mesh")
	FVolDeformTetrahedralMeshParameters TetrahedralMeshParams;
	UPROPERTY(VisibleAnywhere, Transient, Category = "VolDeform")
	TObjectPtr<UInstancedStaticMeshComponent> NodeISMComponent;

	AVolDeformTetrahedralActor(const FObjectInitializer&);
	~AVolDeformTetrahedralActor();

	void PostLoad() override;

#if WITH_EDITOR
	void PostEditChangeProperty(struct FPropertyChangedEvent& PropertyChangedEvent) override;
#endif

	DECLARE_MULTICAST_DELEGATE_OneParam(FBoundingBoxChanged, AVolDeformTetrahedralActor*);
	FBoundingBoxChanged BoundingBoxChanged;

private:
	void setupMesh();

private:
	TObjectPtr<UStaticMesh>		   NodeSM;
	TObjectPtr<UMaterialInterface> NodeMaterial;
};
