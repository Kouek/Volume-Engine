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
	TArray<float>	  TransferFunctionData;
	TArray<float>	  TransferFunctionDataPreIntegrated;
	TArray<glm::vec2> EmptyScalarRanges;

	bool IsComplete() const { return !RAWVolumeData.IsEmpty() && !EmptyScalarRanges.IsEmpty(); }
};

USTRUCT()
struct FVolDataVDBParameters
{
	GENERATED_BODY()

	static constexpr int32 kMaxLevelNum = 3;
	static constexpr int32 kMaxLogChildPerLevel = 9;

	UPROPERTY(VisibleAnywhere)
	EVolDataVoxelType VoxelType = EVolDataVoxelType::None;
	UPROPERTY(VisibleAnywhere)
	int32 RootLevel = 0;
	UPROPERTY(VisibleAnywhere)
	int32 ApronWidth = 1;
	UPROPERTY(VisibleAnywhere)
	int32 ApronAndDepthWidth = 2;
	UPROPERTY(EditAnywhere)
	int32 LogChildPerLevels[kMaxLevelNum] = { 5, 4, 3 };
	int32 LogChildAtLevelZeroCached = 0;
	UPROPERTY(VisibleAnywhere)
	int32 ChildPerLevels[kMaxLevelNum] = { 32, 16, 8 };
	UPROPERTY(VisibleAnywhere)
	int32 ChildCoverVoxelPerLevels[kMaxLevelNum] = { 1, 32, 32 * 16 };
	UPROPERTY(VisibleAnywhere)
	int32 DepthCoordValueInAtlasBrick[2] = { -1, 32 };
	UPROPERTY(VisibleAnywhere)
	int32 VoxelPerAtlasBrick = 34;
	UPROPERTY(VisibleAnywhere)
	FIntVector BrickPerVolume = { 0, 0, 0 };
	UPROPERTY(EditAnywhere)
	uint32 MaxAllowedGPUMemoryInGB = 2;
	UPROPERTY(VisibleAnywhere)
	FIntVector InitialVoxelPerAtlas{ 0, 0, 0 };
	UPROPERTY(VisibleAnywhere)
	FIntVector VoxelPerVolume{ 0, 0, 0 };

	TOptional<FString> InitializeAndCheck(const FIntVector3& InVoxelPerVolume, EVolDataVoxelType InVoxelType);

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

	UPROPERTY(EditAnywhere)
	bool bNeedFullRebuild = true;
	UPROPERTY(VisibleAnywhere)
	uint32 Resolution = 256;
	UPROPERTY(VisibleAnywhere)
	FFilePath SourcePath;
};

UCLASS()
class VOLDATA_API UVolDataVDBComponent : public USceneComponent
{
	GENERATED_BODY()

public:
	UVolDataVDBComponent(const FObjectInitializer&);

	UPROPERTY(EditAnywhere, Category = "VolData")
	FVolDataVDBParameters VDBParams;
	UPROPERTY(VisibleAnywhere, Transient, Category = "VolData")
	UTexture2D* TransferFunction = nullptr;
	UPROPERTY(VisibleAnywhere, Transient, Category = "VolData")
	UTexture2D* TransferFunctionPreIntegrated = nullptr;

	UPROPERTY(EditAnywhere, Category = "VolData")
	FVolDataLoadRAWVolumeParameters LoadRAWVolumeParams;
	UFUNCTION(CallInEditor, Category = "VolData")
	void LoadRAWVolume();

	UPROPERTY(EditAnywhere, Category = "VolData")
	FVolDataLoadTransferFunctionParameters LoadTransferFunctionParameters;
	UFUNCTION(CallInEditor, Category = "VolData")
	void LoadTransferFunction();

	void PostLoad() override;

	std::shared_ptr<DepthBoxVDB::VolData::IVDBBuilder> GetVDBBuilder() const { return VDBBuilder; }
	TSharedPtr<FVolDataVDBCPUData>					   GetCPUData() const { return CPUData; }

	DECLARE_MULTICAST_DELEGATE_OneParam(FOnTransferFunctionChanged, UVolDataVDBComponent*);

	FOnTransferFunctionChanged OnTransferFunctionChanged;

#if WITH_EDITOR
	void PostEditChangeProperty(struct FPropertyChangedEvent& PropertyChangedEvent) override;
#endif

private:
	void setupTransferFunction();
	void buildVDB(bool bNeedReload = false, bool bNeedRelayoutAtlas = false);

private:
	TSharedPtr<FVolDataVDBCPUData> CPUData;

	std::shared_ptr<DepthBoxVDB::VolData::IVDBBuilder> VDBBuilder;
};
