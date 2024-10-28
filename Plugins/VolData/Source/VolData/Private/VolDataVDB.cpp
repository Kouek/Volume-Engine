#include "VolDataVDB.h"

#include "DesktopPlatformModule.h"

void UVolDataVDBComponent::LoadRAWVolume()
{
	FJsonSerializableArray Files;
	FDesktopPlatformModule::Get()->OpenFileDialog(
		FSlateApplication::Get().FindBestParentWindowHandleForDialogs(nullptr), TEXT("Select a RAW Volume file"),
		FPaths::GetProjectFilePath(), TEXT(""), TEXT("Volume|*.raw;*.bin;*.RAW"), EFileDialogFlags::None, Files);
	if (Files.IsEmpty())
		return;

	LoadRAWVolumeParams.SourcePath.FilePath = Files[0];

	BuildVDB(true);
}

void UVolDataVDBComponent::PostLoad()
{
	Super::PostLoad();

	CPUData = MakeShared<FVolDataVDBCPUData>();
	VDBBuilder = DepthBoxVDB ::VolData::IVDBBuilder::Create();

	BuildVDB(true);
}

void UVolDataVDBComponent::BuildVDB(bool bNeedReLoad)
{
	// Temporarily support RAW Volume only
	if (bNeedReLoad)
	{
		auto DataOrErrMsg = FRAWVolumeData::LoadFromFile({ .VoxelType = LoadRAWVolumeParams.VoxelType,
			.VoxelPerVolume = LoadRAWVolumeParams.VoxelPerVolume,
			.AxisOrder = LoadRAWVolumeParams.AxisOrder,
			.SourcePath = LoadRAWVolumeParams.SourcePath });
		if (DataOrErrMsg.IsType<FString>())
		{
			UE_LOG(LogVolData, Log, TEXT("%s"), *DataOrErrMsg.Get<FString>());
			return;
		}

		CPUData->RAWVolumeData = std::move(DataOrErrMsg.Get<TArray<uint8>>());
		CPUData->ReorderedVoxelPerVolume =
			VolData::ReorderVoxelPerVolume(LoadRAWVolumeParams.VoxelPerVolume, LoadRAWVolumeParams.AxisOrder)
				.GetValue()
				.Get<0>();
	}

	{
		auto ErrMsgOpt = VDBParams.InitializeAndCheck(CPUData->ReorderedVoxelPerVolume, LoadRAWVolumeParams.VoxelType);
		if (ErrMsgOpt.IsSet())
		{
			UE_LOG(LogVolData, Log, TEXT("%s"), *ErrMsgOpt.GetValue());
			return;
		}
	}

	AsyncTask(ENamedThreads::Type::AnyThread, [this]() {
		VDBBuilder->FullBuild({ .RAWVolumeData = CPUData->RAWVolumeData.GetData(), .VDBParams = VDBParams });
	});
}

#if WITH_EDITOR
void UVolDataVDBComponent::PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent)
{
	if (PropertyChangedEvent.GetMemberPropertyName() == GET_MEMBER_NAME_CHECKED(UVolDataVDBComponent, VDBParams))
	{
		BuildVDB();
	}
	if (PropertyChangedEvent.GetMemberPropertyName()
		== GET_MEMBER_NAME_CHECKED(UVolDataVDBComponent, LoadRAWVolumeParams))
	{
		BuildVDB(true);
	}
}
#endif
