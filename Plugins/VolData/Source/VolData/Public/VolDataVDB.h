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
	int32 RootLevel;
	UPROPERTY(EditAnywhere)
	int32 ApronWidth = 1;
	UPROPERTY(VisibleAnywhere)
	int32 ApronAndDepthWidth;
	UPROPERTY(EditAnywhere)
	int32 LogChildPerLevels[MaxLevelNum] = { 5, 4, 3 };
	UPROPERTY(VisibleAnywhere)
	int32 ChildPerLevels[MaxLevelNum];
	UPROPERTY(VisibleAnywhere)
	int32 ChildCoverVoxelPerLevels[MaxLevelNum];
	UPROPERTY(VisibleAnywhere)
	int32 DepthPositionInAtlasBrick[2];
	UPROPERTY(EditAnywhere)
	bool bUseDepthBox = true;
	UPROPERTY(VisibleAnywhere)
	int32 VoxelPerAtlasBrick;
	UPROPERTY(VisibleAnywhere)
	FIntVector BrickPerVolume;
	UPROPERTY(VisibleAnywhere)
	int32 MaxAllowedGPUMemoryInGB = 2;
	UPROPERTY(VisibleAnywhere)
	FIntVector InitialVoxelPerAtlas;

	TOptional<FString> InitializeAndCheck(const FIntVector3& VoxelPerVolume, EVolDataVoxelType InVoxelType)
	{
		VoxelType = InVoxelType;

		for (int Lev = 0; Lev < MaxLevelNum; ++Lev)
		{
			int32 LogChildPerLevel = LogChildPerLevels[Lev];
			if (LogChildPerLevel < 0 || LogChildPerLevel > MaxLogChildPerLevel)
				return FString::Format(TEXT("Invalid LogChildPerLevels[{0}] = {1}."), { Lev, LogChildPerLevel });

			ChildPerLevels[Lev] = 1 << LogChildPerLevel;
			ChildCoverVoxelPerLevels[Lev] = Lev == 0 ? 1 : ChildCoverVoxelPerLevels[Lev - 1] * ChildPerLevels[Lev];
		}

		ApronAndDepthWidth = ApronWidth + (bUseDepthBox ? 1 : 0);
		DepthPositionInAtlasBrick[0] = -ApronAndDepthWidth;
		DepthPositionInAtlasBrick[1] = ChildPerLevels[0] - 1 + ApronAndDepthWidth;
		VoxelPerAtlasBrick = ChildPerLevels[0] + ApronAndDepthWidth;

		RootLevel = 0;
		while (true)
		{
			int32 CoverVoxelCurrLevel = ChildCoverVoxelPerLevels[RootLevel] * ChildPerLevels[RootLevel];
			if (CoverVoxelCurrLevel >= VoxelPerVolume.X && CoverVoxelCurrLevel >= VoxelPerVolume.Y
				&& CoverVoxelCurrLevel >= VoxelPerVolume.Z)
				break;

			++RootLevel;
			if (RootLevel == MaxLevelNum)
			{
				return FString::Format(TEXT("VDB cannot cover volume with size {0}."), { VoxelPerVolume.ToString() });
			}
		}

		for (int Dim = 0; Dim < 3; ++Dim)
		{
			BrickPerVolume[Dim] = (VoxelPerVolume[Dim] + ChildPerLevels[0] - 1) / ChildPerLevels[0];
		}

		InitialVoxelPerAtlas = BrickPerVolume * VoxelPerAtlasBrick;
		{
			size_t VoxelSize = VolData::SizeOfVoxelType(VoxelType);
			size_t MaxAllowedGPUMemoryInByte = VoxelSize * MaxAllowedGPUMemoryInGB * (1 << 30);
			while ([&]() {
				return InitialVoxelPerAtlas.Z > 0
					&& VoxelSize * InitialVoxelPerAtlas.X * InitialVoxelPerAtlas.Y * InitialVoxelPerAtlas.Z
					> MaxAllowedGPUMemoryInByte;
			}())
				InitialVoxelPerAtlas.Z -= ApronAndDepthWidth;
		}

		if (InitialVoxelPerAtlas.Z < VoxelPerVolume.Z)
		{
			return FString(
				TEXT("Volume texture streaming is NOT supported yet! MaxAllowedGPUMemoryInGB is too small."));
		}

		return {};
	}

	operator DepthBoxVDB::VolData::VDBParameters()
	{
		DepthBoxVDB::VolData::VDBParameters Ret;
#define ASSIGN(Member) Ret.Member = Member

		Ret.VoxelType = (DepthBoxVDB::VolData::EVoxelType)(uint8)VoxelType;
		ASSIGN(RootLevel);
		ASSIGN(ApronWidth);
		ASSIGN(ApronAndDepthWidth);
		for (int32 i = 0; i < MaxLevelNum; ++i)
		{
			ASSIGN(LogChildPerLevels[i]);
			ASSIGN(ChildPerLevels[i]);
			ASSIGN(ChildCoverVoxelPerLevels[i]);
		}
		ASSIGN(DepthPositionInAtlasBrick[0]);
		ASSIGN(DepthPositionInAtlasBrick[1]);
		ASSIGN(bUseDepthBox);
		ASSIGN(VoxelPerAtlasBrick);
		for (int32 i = 0; i < 3; ++i)
		{
			ASSIGN(BrickPerVolume[i]);
			ASSIGN(InitialVoxelPerAtlas[i]);
		}

#undef ASSIGN

		return Ret;
	}
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

	void BuildVDB(bool bNeedReLoad = false);

#if WITH_EDITOR
	void PostEditChangeProperty(struct FPropertyChangedEvent& PropertyChangedEvent) override;
#endif

private:
	TSharedPtr<FVolDataVDBCPUData> CPUData;

	std::unique_ptr<DepthBoxVDB::VolData::IVDBBuilder> VDBBuilder;
};
