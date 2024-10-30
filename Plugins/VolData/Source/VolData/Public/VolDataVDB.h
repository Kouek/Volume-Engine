#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"

#include "VolDataUtil.h"

#include "DepthBoxVDB/VolData.h"

#include "VolDataVDB.generated.h"

struct FVolDataVDBCPUData
{
	FIntVector		  VoxelPerVolume;
	uint32			  EmptyScalarRangeNum;
	TArray<uint8>	  RAWVolumeData;
	TArray<glm::vec2> EmptyScalarRanges;

	bool IsComplete() const { return !RAWVolumeData.IsEmpty() && !EmptyScalarRanges.IsEmpty(); }
};

USTRUCT()
struct FVolDataVDBParameters
{
	GENERATED_BODY()

	static constexpr int32 MaxLevelNum = 3;
	static constexpr int32 MaxLogChildPerLevel = 9;

	UPROPERTY(VisibleAnywhere)
	EVolDataVoxelType VoxelType;
	UPROPERTY(VisibleAnywhere)
	int32 RootLevel = 0;
	UPROPERTY(EditAnywhere)
	int32 ApronWidth = 1;
	UPROPERTY(VisibleAnywhere)
	int32 ApronAndDepthWidth = 1;
	UPROPERTY(EditAnywhere)
	int32 LogChildPerLevels[MaxLevelNum] = { 5, 4, 3 };
	int32 LogChildAtLevelZeroCached = 0;
	UPROPERTY(VisibleAnywhere)
	int32 ChildPerLevels[MaxLevelNum] = { 32, 16, 8 };
	UPROPERTY(VisibleAnywhere)
	int32 ChildCoverVoxelPerLevels[MaxLevelNum] = { 1, 32, 32 * 16 };
	UPROPERTY(VisibleAnywhere)
	int32 DepthPositionInAtlasBrick[2] = { -1, 32 };
	UPROPERTY(EditAnywhere)
	bool bUseDepthBox = true;
	UPROPERTY(VisibleAnywhere)
	int32 VoxelPerAtlasBrick = 34;
	UPROPERTY(VisibleAnywhere)
	FIntVector BrickPerVolume = { 0, 0, 0 };
	UPROPERTY(EditAnywhere)
	int32 MaxAllowedGPUMemoryInGB = 2;
	UPROPERTY(VisibleAnywhere)
	FIntVector InitialVoxelPerAtlas{ 0, 0, 0 };

	TOptional<FString> InitializeAndCheck(const FIntVector3& VoxelPerVolume, EVolDataVoxelType InVoxelType);

	operator DepthBoxVDB::VolData::VDBParameters();
};

USTRUCT()
struct FVolDataLoadRAWVolumeParameters
{
	GENERATED_BODY()

	bool bNeedReload = false;

	UPROPERTY(EditAnywhere)
	EVolDataVoxelType VoxelType = EVolDataVoxelType::None;
	UPROPERTY(EditAnywhere)
	FIntVector AxisOrder = { 1, 2, 3 };
	UPROPERTY(EditAnywhere)
	FIntVector VoxelPerVolume = { 256, 256, 256 };
	UPROPERTY(VisibleAnywhere)
	FFilePath SourcePath;
};

USTRUCT()
struct FVolDataLoadTransferFunctionParameters
{
	GENERATED_BODY()

	bool bNeedReload = false;

	UPROPERTY(VisibleAnywhere)
	uint32 Resolution = 256;
	UPROPERTY(VisibleAnywhere)
	FFilePath SourcePath;
};

UCLASS()
class VOLDATA_API UVolDataVDBComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	UPROPERTY(EditAnywhere, Category = "VolData")
	FVolDataVDBParameters VDBParams;

	UPROPERTY(EditAnywhere, Category = "VolData")
	FVolDataLoadRAWVolumeParameters LoadRAWVolumeParams;
	UFUNCTION(CallInEditor, Category = "VolData")
	void LoadRAWVolume();

	UPROPERTY(EditAnywhere, Category = "VolData")
	FVolDataLoadTransferFunctionParameters LoadTransferFunctionParameters;
	UFUNCTION(CallInEditor, Category = "VolData")
	void LoadTransferFunction();

	void PostLoad() override;

	void BuildVDB(bool bNeedReload = false, bool bNeedRelayoutAtlas = false);

#if WITH_EDITOR
	void PostEditChangeProperty(struct FPropertyChangedEvent& PropertyChangedEvent) override;
#endif

private:
	TSharedPtr<FVolDataVDBCPUData> CPUData;

	std::shared_ptr<DepthBoxVDB::VolData::IVDBDataProvider> VDBDataProvider;
	std::unique_ptr<DepthBoxVDB::VolData::IVDBBuilder>		VDBBuilder;
};
