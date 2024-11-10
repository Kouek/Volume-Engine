#include "VolDataVDB.h"

#include "DesktopPlatformModule.h"

#include "VolDataRAWVolume.h"
#include "VolDataTransferFunction.h"

TOptional<FString> FVolDataVDBParameters::InitializeAndCheck(
	const FIntVector3& InVoxelPerVolume, EVolDataVoxelType InVoxelType)
{
	VoxelType = InVoxelType;
	VoxelPerVolume = InVoxelPerVolume;

#define CHECK(Member, Min, Max)                                                 \
	if (Member < Min || Member > Max)                                           \
	{                                                                           \
		return FString::Format(TEXT("Invalid " #Member " = {0}."), { Member }); \
	}

	for (int32 i = 0; i < MaxLevelNum; ++i)
	{
		CHECK(LogChildPerLevels[i], 1, MaxLogChildPerLevel);
	}
	CHECK(MaxAllowedGPUMemoryInGB, 1, 64);

#undef CHECK

	LogChildAtLevelZeroCached = LogChildPerLevels[0];

	for (int Lev = 0; Lev < MaxLevelNum; ++Lev)
	{
		int32 LogChildPerLevel = LogChildPerLevels[Lev];
		if (LogChildPerLevel < 0 || LogChildPerLevel > MaxLogChildPerLevel)
			return FString::Format(TEXT("Invalid LogChildPerLevels[{0}] = {1}."), { Lev, LogChildPerLevel });

		ChildPerLevels[Lev] = 1 << LogChildPerLevel;
		ChildCoverVoxelPerLevels[Lev] = Lev == 0 ? 1 : ChildCoverVoxelPerLevels[Lev - 1] * ChildPerLevels[Lev - 1];
	}

	DepthCoordValueInAtlasBrick[0] = -ApronAndDepthWidth;
	DepthCoordValueInAtlasBrick[1] = ChildPerLevels[0] - 1 + ApronAndDepthWidth;
	VoxelPerAtlasBrick = ChildPerLevels[0] + 2 * ApronAndDepthWidth;

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

	for (int32 Axis = 0; Axis < 3; ++Axis)
	{
		BrickPerVolume[Axis] = (VoxelPerVolume[Axis] + ChildPerLevels[0] - 1) / ChildPerLevels[0];
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
			InitialVoxelPerAtlas.Z -= VoxelPerAtlasBrick;
	}

	if (InitialVoxelPerAtlas.Z == 0)
	{
		return FString(
			TEXT("MaxAllowedGPUMemoryInGB is too small. VoxelPerAtlas.x * VoxelPerAtlas.y cannot be contained."));
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
	ASSIGN(DepthCoordValueInAtlasBrick[0]);
	ASSIGN(DepthCoordValueInAtlasBrick[1]);
	ASSIGN(VoxelPerAtlasBrick);
	for (int32 i = 0; i < 3; ++i)
	{
		ASSIGN(BrickPerVolume[i]);
		ASSIGN(VoxelPerVolume[i]);
	}

#undef ASSIGN

	return Ret;
}

UVolDataVDBComponent::UVolDataVDBComponent(const FObjectInitializer&)
{
	CPUData = MakeShared<FVolDataVDBCPUData>();
	VDBBuilder = DepthBoxVDB::VolData::IVDBBuilder::Create({});
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

	buildVDB();
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

	setupTransferFunction();
	buildVDB();
}

void UVolDataVDBComponent::PostLoad()
{
	Super::PostLoad();

	setupTransferFunction(true);
	buildVDB(true);
}

#if WITH_EDITOR
void UVolDataVDBComponent::PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent)
{
	if (PropertyChangedEvent.GetMemberPropertyName() == GET_MEMBER_NAME_CHECKED(UVolDataVDBComponent, VDBParams))
	{
		if (PropertyChangedEvent.GetPropertyName() == GET_MEMBER_NAME_CHECKED(FVolDataVDBParameters, ChildPerLevels))
		{
			buildVDB(false, VDBParams.LogChildAtLevelZeroCached != VDBParams.LogChildPerLevels[0]);
		}
		else
		{
			buildVDB();
		}
	}
}
#endif

void UVolDataVDBComponent::setupTransferFunction(bool bNeedReload)
{
	if (!bNeedReload && !LoadTransferFunctionParameters.bNeedReload)
		return;

	CPUData->EmptyScalarRanges.Empty();

	auto DataOrErrMsg = FVolDataTransferFunction::LoadFromFile(
		FVolDataTransferFunction::LoadFromFileParameters<true>{ .Resolution = LoadTransferFunctionParameters.Resolution,
			.SourcePath = LoadTransferFunctionParameters.SourcePath });
	if (DataOrErrMsg.IsType<FString>())
	{
		UE_LOG(LogVolData, Error, TEXT("%s"), *DataOrErrMsg.Get<FString>());
		return;
	}
	auto& Data = DataOrErrMsg.Get<TArray<FFloat16>>();

	{
		uint32_t EmptyRange[2] = { LoadTransferFunctionParameters.Resolution, 0 };
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
	}

	auto TexOrErrMsg = FVolDataTransferFunction::CreateTexture(
		{ .Resolution = LoadTransferFunctionParameters.Resolution, .TFFlatArray = Data });
	if (TexOrErrMsg.IsType<FString>())
	{
		UE_LOG(LogVolData, Error, TEXT("%s"), *TexOrErrMsg.Get<FString>());
		return;
	}
	TransferFunction = TexOrErrMsg.Get<UTexture2D*>();

	TexOrErrMsg = FVolDataTransferFunction::CreateTexturePreIntegrated(
		{ .Resolution = LoadTransferFunctionParameters.Resolution, .TFFlatArray = Data });
	if (TexOrErrMsg.IsType<FString>())
	{
		UE_LOG(LogVolData, Error, TEXT("%s"), *TexOrErrMsg.Get<FString>());
		return;
	}
	TransferFunctionPreIntegrated = TexOrErrMsg.Get<UTexture2D*>();

	CPUData->TransferFunctionData =
		FVolDataTransferFunction::LoadFromFile(FVolDataTransferFunction::LoadFromFileParameters<false>{
												   .Resolution = LoadTransferFunctionParameters.Resolution,
												   .SourcePath = LoadTransferFunctionParameters.SourcePath })
			.Get<TArray<float>>();
	CPUData->TransferFunctionDataPreIntegrated = FVolDataTransferFunction::PreIntegrateFromFlatArray<false>(
		CPUData->TransferFunctionData, LoadTransferFunctionParameters.Resolution);

	LoadTransferFunctionParameters.bNeedReload = false;
	OnTransferFunctionChanged.Broadcast(this);
}

void UVolDataVDBComponent::buildVDB(bool bNeedReload, bool bNeedRelayoutAtlas)
{
	// Temporarily support RAW Volume only
	bool bNeedRecreateVDBProvider = bNeedRelayoutAtlas;
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
		bNeedRecreateVDBProvider = true;
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

	AsyncTask(ENamedThreads::Type::AnyThread, [this, bNeedRecreateVDBProvider]() {
		VolData::FStdOutputLinker Linker;
		if (bNeedRecreateVDBProvider)
		{
			VDBDataProvider =
				DepthBoxVDB::VolData::IVDBDataProvider::Create({ .RAWVolumeData = CPUData->RAWVolumeData.GetData(),
					.EmptyScalarRanges = CPUData->EmptyScalarRanges.GetData(),
					.EmptyScalarRangeNum = CPUData->EmptyScalarRangeNum,
					.MaxAllowedGPUMemoryInGB = VDBParams.MaxAllowedGPUMemoryInGB,
					.VDBParams = VDBParams });
		}
		VDBBuilder->FullBuild({ .EmptyScalarRangeNum = CPUData->EmptyScalarRangeNum,
			.EmptyScalarRanges = CPUData->EmptyScalarRanges.GetData(),
			.Provider = VDBDataProvider });
	});
}
