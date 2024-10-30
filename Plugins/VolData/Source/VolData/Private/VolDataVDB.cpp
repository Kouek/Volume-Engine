#include "VolDataVDB.h"

#include "DesktopPlatformModule.h"

#include "VolDataRAWVolume.h"
#include "VolDataTransferFunction.h"

TOptional<FString> FVolDataVDBParameters::InitializeAndCheck(
	const FIntVector3& VoxelPerVolume, EVolDataVoxelType InVoxelType)
{
	VoxelType = InVoxelType;

	LogChildAtLevelZeroCached = LogChildPerLevels[0];

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
		return FString(TEXT("Volume texture streaming is NOT supported yet! MaxAllowedGPUMemoryInGB is too small."));
	}

	return {};
}

FVolDataVDBParameters::operator DepthBoxVDB::VolData::VDBParameters()
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

void UVolDataVDBComponent::LoadRAWVolume()
{
	FJsonSerializableArray Files;
	FDesktopPlatformModule::Get()->OpenFileDialog(
		FSlateApplication::Get().FindBestParentWindowHandleForDialogs(nullptr), TEXT("Select a RAW Volume file"),
		FPaths::GetProjectFilePath(), TEXT(""), TEXT("Volume|*.raw;*.bin;*.RAW"), EFileDialogFlags::None, Files);
	if (Files.IsEmpty())
		return;

	LoadRAWVolumeParams.SourcePath.FilePath = Files[0];
	LoadRAWVolumeParams.bNeedReload = true;

	BuildVDB();
}

void UVolDataVDBComponent::LoadTransferFunction()
{
	FJsonSerializableArray Files;
	FDesktopPlatformModule::Get()->OpenFileDialog(
		FSlateApplication::Get().FindBestParentWindowHandleForDialogs(nullptr), TEXT("Select a Transfer Function file"),
		FPaths::GetProjectFilePath(), TEXT(""), TEXT("Volume|*.txt"), EFileDialogFlags::None, Files);
	if (Files.IsEmpty())
		return;

	LoadTransferFunctionParameters.SourcePath.FilePath = Files[0];
	LoadTransferFunctionParameters.bNeedReload = true;

	BuildVDB();
}

void UVolDataVDBComponent::PostLoad()
{
	Super::PostLoad();

	CPUData = MakeShared<FVolDataVDBCPUData>();
	VDBBuilder = DepthBoxVDB::VolData::IVDBBuilder::Create({});
	VDBDataProvider = DepthBoxVDB::VolData::IVDBDataProvider::Create({});

	BuildVDB(true);
}

void UVolDataVDBComponent::BuildVDB(bool bNeedReload, bool bNeedRelayoutAtlas)
{
	// Temporarily support RAW Volume only
	bool bNeedTransferRAWVolumeToAtlas = bNeedRelayoutAtlas;
	if (bNeedReload || LoadRAWVolumeParams.bNeedReload)
	{
		CPUData->RAWVolumeData.Empty();

		auto DataOrErrMsg = FVolDataRAWVolumeData::LoadFromFile({ .VoxelType = LoadRAWVolumeParams.VoxelType,
			.VoxelPerVolume = LoadRAWVolumeParams.VoxelPerVolume,
			.AxisOrder = LoadRAWVolumeParams.AxisOrder,
			.SourcePath = LoadRAWVolumeParams.SourcePath });
		if (DataOrErrMsg.IsType<FString>())
		{
			UE_LOG(LogVolData, Error, TEXT("%s"), *DataOrErrMsg.Get<FString>());
			return;
		}

		CPUData->RAWVolumeData = std::move(DataOrErrMsg.Get<TArray<uint8>>());
		CPUData->VoxelPerVolume =
			VolData::ReorderVoxelPerVolume(LoadRAWVolumeParams.VoxelPerVolume, LoadRAWVolumeParams.AxisOrder)
				.GetValue()
				.Get<1>();

		LoadRAWVolumeParams.bNeedReload = false;
		bNeedTransferRAWVolumeToAtlas = true;
	}

	if (bNeedReload || LoadTransferFunctionParameters.bNeedReload)
	{
		CPUData->EmptyScalarRanges.Empty();

		auto DataOrErrMsg =
			FVolDataTransferFunction::LoadFromFile({ .Resolution = LoadTransferFunctionParameters.Resolution,
				.SourcePath = LoadTransferFunctionParameters.SourcePath });
		if (DataOrErrMsg.IsType<FString>())
		{
			UE_LOG(LogVolData, Error, TEXT("%s"), *DataOrErrMsg.Get<FString>());
			return;
		}

		uint32_t EmptyRange[2] = { LoadTransferFunctionParameters.Resolution, 0 };
		auto&	 Data = DataOrErrMsg.Get<TArray<float>>();
		CPUData->EmptyScalarRanges.Empty();
		CPUData->EmptyScalarRangeNum = 0;
		auto Append = [&]() {
			CPUData->EmptyScalarRanges.Add(glm::vec2(EmptyRange[0], EmptyRange[1]));
			++CPUData->EmptyScalarRangeNum;
		};
		for (uint32 Scalar = 0; Scalar < LoadTransferFunctionParameters.Resolution; ++Scalar)
		{
			bool bCurrEmpty = Data[Scalar] <= std::numeric_limits<float>::epsilon();
			if (bCurrEmpty)
			{
				EmptyRange[0] = std::min(Scalar, EmptyRange[0]);
				EmptyRange[1] = std::max(Scalar, EmptyRange[1]);
			}
			else if (EmptyRange[0] < EmptyRange[1])
			{
				Append();
				EmptyRange[0] = LoadTransferFunctionParameters.Resolution;
				EmptyRange[1] = 0;
			}
		}
		if (EmptyRange[0] < EmptyRange[1])
		{
			Append();
		}

		LoadTransferFunctionParameters.bNeedReload = false;
	}

	if (!CPUData->IsComplete())
	{
		UE_LOG(LogVolData, Error, TEXT("CPUData is incomplete, cannot perform VDB Building."));
		return;
	}

	{
		auto ErrMsgOpt = VDBParams.InitializeAndCheck(CPUData->VoxelPerVolume, LoadRAWVolumeParams.VoxelType);
		if (ErrMsgOpt.IsSet())
		{
			UE_LOG(LogVolData, Error, TEXT("%s"), *ErrMsgOpt.GetValue());
			return;
		}
	}

	AsyncTask(ENamedThreads::Type::AnyThread, [this, bNeedTransferRAWVolumeToAtlas]() {
		VolData::FStdOutputLinker Linker;
		if (bNeedTransferRAWVolumeToAtlas)
		{
			VDBDataProvider->TransferRAWVolumeToAtlas({ .VoxelPerVolume = glm::uvec3(CPUData->VoxelPerVolume.X,
															CPUData->VoxelPerVolume.Y, CPUData->VoxelPerVolume.Z),
				.RAWVolumeData = CPUData->RAWVolumeData.GetData(),
				.VDBParams = VDBParams });
		}
		VDBBuilder->FullBuild({ .EmptyScalarRangeNum = CPUData->EmptyScalarRangeNum,
			.EmptyScalarRanges = CPUData->EmptyScalarRanges.GetData(),
			.Provider = VDBDataProvider,
			.VDBParams = VDBParams });
	});
}

#if WITH_EDITOR
void UVolDataVDBComponent::PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent)
{
	if (PropertyChangedEvent.GetMemberPropertyName() == GET_MEMBER_NAME_CHECKED(UVolDataVDBComponent, VDBParams))
	{
		if (PropertyChangedEvent.GetPropertyName() == GET_MEMBER_NAME_CHECKED(FVolDataVDBParameters, ChildPerLevels))
		{
			BuildVDB(false, VDBParams.LogChildAtLevelZeroCached != VDBParams.LogChildPerLevels[0]);
		}
		else
		{
			BuildVDB();
		}
	}
}
#endif
