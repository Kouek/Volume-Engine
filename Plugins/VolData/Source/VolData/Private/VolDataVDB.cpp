#include "VolDataVDB.h"

#include "DesktopPlatformModule.h"

#include "VolDataRAWVolume.h"
#include "VolDataTransferFunction.h"

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

	BuildVDB(true);
}

void UVolDataVDBComponent::BuildVDB(bool bNeedReLoad)
{
	// Temporarily support RAW Volume only
	if (bNeedReLoad || LoadRAWVolumeParams.bNeedReload)
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
	}

	if (bNeedReLoad || LoadTransferFunctionParameters.bNeedReload)
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

	AsyncTask(ENamedThreads::Type::AnyThread, [this]() {
		VolData::FStdOutputLinker Linker;
		VDBBuilder->FullBuild({ .VoxelPerVolume = glm::uvec3(
									CPUData->VoxelPerVolume.X, CPUData->VoxelPerVolume.Y, CPUData->VoxelPerVolume.Z),
			.EmptyScalarRangeNum = CPUData->EmptyScalarRangeNum,
			.RAWVolumeData = CPUData->RAWVolumeData.GetData(),
			.EmptyScalarRanges = CPUData->EmptyScalarRanges.GetData(),
			.VDBParams = VDBParams });
	});
}

#if WITH_EDITOR
void UVolDataVDBComponent::PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent)
{
	if (PropertyChangedEvent.GetMemberPropertyName() == GET_MEMBER_NAME_CHECKED(UVolDataVDBComponent, VDBParams))
	{
		BuildVDB();
	}
	else if (PropertyChangedEvent.GetMemberPropertyName()
		== GET_MEMBER_NAME_CHECKED(UVolDataVDBComponent, LoadRAWVolumeParams))
	{
		LoadRAWVolumeParams.bNeedReload = true;
		BuildVDB();
	}
	else if (PropertyChangedEvent.GetMemberPropertyName()
		== GET_MEMBER_NAME_CHECKED(UVolDataVDBComponent, LoadTransferFunctionParameters))
	{
		LoadTransferFunctionParameters.bNeedReload = true;
		BuildVDB();
	}
}
#endif
