#include "VolDeformTetrahedralActor.h"

TOptional<FString> FVolDeformTetrahedralMeshParameters::InitializeAndCheck()
{
#define CHECK(Member, Min, Max)                                                 \
	if (Member < Min || Member > Max)                                           \
	{                                                                           \
		return FString::Format(TEXT("Invalid " #Member " = {0}."), { Member }); \
	}

	for (int32 Axis = 0; Axis < 3; ++Axis)
	{
		CHECK(Resolution[Axis], 1, 100);
		CHECK(Extent[Axis], 1.f, 1000.f);
	}

#undef CHECK

	return {};
}

AVolDeformTetrahedralActor::AVolDeformTetrahedralActor(const FObjectInitializer&)
{
	{
		NodeISMComponent = CreateDefaultSubobject<UInstancedStaticMeshComponent>(TEXT("Node ISM"));
		NodeISMComponent->bHasPerInstanceHitProxies = true;
		SetRootComponent(NodeISMComponent);

		NodeISMComponent->TransformUpdated.AddLambda(
			[this](USceneComponent*, EUpdateTransformFlags, ETeleportType) { BoundingBoxChanged.Broadcast(this); });
	}
}

AVolDeformTetrahedralActor::~AVolDeformTetrahedralActor() {}

void AVolDeformTetrahedralActor::PostLoad()
{
	AActor::PostLoad();

	NodeSM = Cast<UStaticMesh>(
		StaticLoadObject(UStaticMesh::StaticClass(), nullptr, TEXT("StaticMesh'/Engine/BasicShapes/Sphere.Sphere'")));
	NodeMaterial = Cast<UMaterialInterface>(StaticLoadObject(
		UMaterial::StaticClass(), nullptr, TEXT("Material'/Engine/MapTemplates/Materials/BasicAsset01.BasicAsset01'")));
	setupMesh();
}

#if WITH_EDITOR
void AVolDeformTetrahedralActor::PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent)
{
	Super::PostEditChangeProperty(PropertyChangedEvent);

	if (PropertyChangedEvent.GetMemberPropertyName()
		== GET_MEMBER_NAME_CHECKED(AVolDeformTetrahedralActor, TetrahedralMeshParams))
	{
		setupMesh();
	}
}
#endif

void AVolDeformTetrahedralActor::setupMesh()
{
	auto ErrOpt = TetrahedralMeshParams.InitializeAndCheck();
	if (ErrOpt.IsSet())
	{
		UE_LOG(LogVolDeform, Log, TEXT("%s"), *ErrOpt.GetValue());
		return;
	}

	{
		if (!NodeISMComponent->GetStaticMesh())
		{
			NodeISMComponent->SetStaticMesh(NodeSM);
			NodeISMComponent->SetMaterial(0, NodeMaterial);
		}
		NodeISMComponent->ClearInstances();

		FVector	   MaxCoord(TetrahedralMeshParams.Resolution);
		FTransform Transform;
		FIntVector Coord;
		for (Coord.Z = 0; Coord.Z <= TetrahedralMeshParams.Resolution.Z; ++Coord.Z)
			for (Coord.Y = 0; Coord.Y <= TetrahedralMeshParams.Resolution.Y; ++Coord.Y)
				for (Coord.X = 0; Coord.X <= TetrahedralMeshParams.Resolution.X; ++Coord.X)
				{
					Transform.SetScale3D(FVector(TetrahedralMeshParams.NodeScale));
					Transform.SetTranslation(TetrahedralMeshParams.Extent * FVector(Coord) / MaxCoord);
					NodeISMComponent->AddInstance(Transform);
				}
	}

	BoundingBoxChanged.Broadcast(this);
}
