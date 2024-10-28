#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"

#include "VolDataUtil.h"

#include "DepthBoxVDB/VolData.h"

#include "VolDataVDB.generated.h"

struct FVolDataVDBCPUData
{
	FIntVector	  ReorderedVoxelPerVolume;
	TArray<uint8> RAWVolumeData;
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
	UPROPERTY(EditAnywhere)
	FIntVector InitialVoxelPerAtlas;

	TOptional<FString> InitializeAndCheck(const FIntVector3& ReorderedVoxelPerVolume, EVolDataVoxelType InVoxelType)
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
			if (ChildCoverVoxelPerLevels[RootLevel] >= ReorderedVoxelPerVolume.X
				&& ChildCoverVoxelPerLevels[RootLevel] >= ReorderedVoxelPerVolume.Y
				&& ChildCoverVoxelPerLevels[RootLevel] >= ReorderedVoxelPerVolume.Z)
				break;

			++RootLevel;
			if (RootLevel == MaxLevelNum)
			{
				return FString::Format(
					TEXT("VDB cannot cover volume with size {0}."), { ReorderedVoxelPerVolume.ToString() });
			}
		}

		for (int Dim = 0; Dim < 3; ++Dim)
		{
			BrickPerVolume[Dim] = (ReorderedVoxelPerVolume[Dim] + ChildPerLevels[0] - 1) / ChildCoverVoxelPerLevels[0];
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

	UPROPERTY(EditAnywhere)
	EVolDataVoxelType VoxelType = EVolDataVoxelType::None;
	UPROPERTY(EditAnywhere)
	FIntVector AxisOrder = { 1, 2, 3 };
	UPROPERTY(EditAnywhere)
	FIntVector VoxelPerVolume = { 256, 256, 256 };
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

	void PostLoad() override;

	void BuildVDB(bool bNeedReLoad = false);

#if WITH_EDITOR
	void PostEditChangeProperty(struct FPropertyChangedEvent& PropertyChangedEvent) override;
#endif

private:
	TSharedPtr<FVolDataVDBCPUData> CPUData;

	std::unique_ptr<DepthBoxVDB::VolData::IVDBBuilder> VDBBuilder;
};
