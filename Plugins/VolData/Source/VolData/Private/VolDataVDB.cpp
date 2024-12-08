#include "VolDataVDB.h"

#include <array>
#include <tuple>

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

	for (int32 i = 0; i < kMaxLevelNum; ++i)
	{
		CHECK(LogChildPerLevels[i], 2, kMaxLogChildPerLevel);
	}
	CHECK(MaxAllowedGPUMemoryInGB, 1, 64);
	CHECK(MaxAllowedResidentFrameNum, 1, 6);

#undef CHECK

	LogChildAtLevelZeroCached = LogChildPerLevels[0];

	for (int Lev = 0; Lev < kMaxLevelNum; ++Lev)
	{
		int32 LogChildPerLevel = LogChildPerLevels[Lev];
		if (LogChildPerLevel < 0 || LogChildPerLevel > kMaxLogChildPerLevel)
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
		if (RootLevel == kMaxLevelNum)
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

	Ret.VoxelType = (DepthBoxVDB::EVoxelType)(uint8)VoxelType;
	ASSIGN(RootLevel);
	ASSIGN(ApronWidth);
	ASSIGN(ApronAndDepthWidth);
	for (int32 i = 0; i < kMaxLevelNum; ++i)
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
	TransferFunctionCurve = CreateDefaultSubobject<UCurveLinearColor>(TEXT("TF Curve"));
	{
		std::array<FRichCurve*, 4> Curves{ &TransferFunctionCurve->FloatCurves[0],
			&TransferFunctionCurve->FloatCurves[1], &TransferFunctionCurve->FloatCurves[2],
			&TransferFunctionCurve->FloatCurves[3] };

		using RGBAType = std::array<float, 4>;
		std::array<std::tuple<float, RGBAType>, 3> InitialPoints{ std::make_tuple(0.f, RGBAType{ 0.f, 0.f, 0.f, 0.f }),
			std::make_tuple(30.f, RGBAType{ 0.f, 0.f, 0.f, 0.f }),
			std::make_tuple(255.f, RGBAType{ 1.f, .5f, .2f, 1.f }) };
		for (auto& [Scalar, RGBA] : InitialPoints)
			for (int32 Cmpt = 0; Cmpt < 4; ++Cmpt)
			{
				Curves[Cmpt]->AddKey(Scalar, RGBA[Cmpt]);
			}

		TransferFunctionCurve->OnUpdateCurve.AddLambda([this](UCurveBase* Curve, EPropertyChangeType::Type ChangeType) {
			std::array<FRichCurve*, 4> Curves{ &TransferFunctionCurve->FloatCurves[0],
				&TransferFunctionCurve->FloatCurves[1], &TransferFunctionCurve->FloatCurves[2],
				&TransferFunctionCurve->FloatCurves[3] };
			TMap<FKeyHandle, uint8>	   KeysOutOfRange;
			for (int32 Cmpt = 0; Cmpt < 4; ++Cmpt)
			{
				KeysOutOfRange.Empty();
				auto KeyHandleItr = Curves[Cmpt]->GetKeyHandleIterator();
				auto KeyItr = Curves[Cmpt]->GetKeyIterator();
				while (KeyItr)
				{
					if (KeyItr->Time < 0.f)
						KeysOutOfRange.Emplace(*KeyHandleItr, 0);
					else if (KeyItr->Time > LoadTransferFunctionParameters.MaxScalarInTF)
						KeysOutOfRange.Emplace(*KeyHandleItr, 1);

					++KeyItr;
					++KeyHandleItr;
				}

				for (auto& [Handle, State] : KeysOutOfRange)
				{
					if (State == 0)
						Curves[Cmpt]->SetKeyTime(Handle, 0.f);
					else
						Curves[Cmpt]->SetKeyTime(Handle, LoadTransferFunctionParameters.MaxScalarInTF);
				}
			}

			syncTransferFunctionFromCurve();
		});
	}

	CPUData = MakeShared<FVolDataVDBCPUData>();
	VDB = DepthBoxVDB::VolData::IVDB::Create({});
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

	buildVDB();
}

void UVolDataVDBComponent::FullRebuildVDB()
{
	buildVDB(false, true);
}

void UVolDataVDBComponent::PostLoad()
{
	Super::PostLoad();

	buildVDB(true);
}

#if WITH_EDITOR
void UVolDataVDBComponent::PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent)
{
	Super::PostEditChangeProperty(PropertyChangedEvent);

	if (PropertyChangedEvent.GetMemberPropertyName() == GET_MEMBER_NAME_CHECKED(UVolDataVDBComponent, VDBParams))
	{
		if (PropertyChangedEvent.GetPropertyName() == GET_MEMBER_NAME_CHECKED(FVolDataVDBParameters, LogChildPerLevels))
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

void UVolDataVDBComponent::loadRAWVolume()
{
	// Temporarily support RAW Volume only
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
}

void UVolDataVDBComponent::loadTransferFunction()
{
	// Load from file
	auto DataOrErrMsg =
		FVolDataTransferFunction::LoadFromFile({ .SourcePath = LoadTransferFunctionParameters.SourcePath });
	if (DataOrErrMsg.IsType<FString>())
	{
		UE_LOG(LogVolData, Error, TEXT("%s"), *DataOrErrMsg.Get<FString>());
		return;
	}

	// Setup Curve
	auto& Data = DataOrErrMsg.Get<TMap<float, FVector4f>>();
	LoadTransferFunctionParameters.MaxScalarInTF = 0.f;
	{
		TransferFunctionCurve->ResetCurve();
		std::array<FRichCurve*, 4> Curves{ &TransferFunctionCurve->FloatCurves[0],
			&TransferFunctionCurve->FloatCurves[1], &TransferFunctionCurve->FloatCurves[2],
			&TransferFunctionCurve->FloatCurves[3] };

		for (auto& [Scalar, RGBA] : Data)
			for (int32 Cmpt = 0; Cmpt < 4; ++Cmpt)
			{
				Curves[Cmpt]->AddKey(Scalar, RGBA[Cmpt]);
				LoadTransferFunctionParameters.MaxScalarInTF =
					FMath::Max(LoadTransferFunctionParameters.MaxScalarInTF, Scalar);
			}
	}

	// Get Empty Scalar Ranges
	{
		std::array<float, 2> EmptyRange{ LoadTransferFunctionParameters.MaxScalarInTF, 0 };
		CPUData->EmptyScalarRanges.Empty();
		CPUData->EmptyScalarRangeNum = 0;
		auto Append = [&]() {
			CPUData->EmptyScalarRanges.Add(glm::vec2(EmptyRange[0], EmptyRange[1]));
			++CPUData->EmptyScalarRangeNum;
		};
		for (auto& [Scalar, RGBA] : Data)
		{
			bool bCurrEmpty = RGBA.W <= std::numeric_limits<float>::epsilon();
			if (bCurrEmpty)
			{
				EmptyRange[0] = std::min(Scalar, EmptyRange[0]);
				EmptyRange[1] = std::max(Scalar, EmptyRange[1]);
			}
			else if (EmptyRange[0] < EmptyRange[1])
			{
				Append();
				EmptyRange[0] = LoadTransferFunctionParameters.MaxScalarInTF;
				EmptyRange[1] = 0;
			}
		}
		if (EmptyRange[0] < EmptyRange[1])
		{
			Append();
		}
	}
}

void UVolDataVDBComponent::syncTransferFunctionFromCurve()
{
	// Get flatten data from Curve
	TArray<FFloat16> FlattenDataHalfPrecision;
	TArray<float>	 FlattenDataFullPrecision;
	{
		FlattenDataHalfPrecision.Reserve(4 * LoadTransferFunctionParameters.Resolution);
		FlattenDataFullPrecision.Reserve(4 * LoadTransferFunctionParameters.Resolution);
		float KeyIdxToScalar =
			LoadTransferFunctionParameters.MaxScalarInTF / (LoadTransferFunctionParameters.Resolution - 1);
		for (uint32 KeyIdx = 0; KeyIdx < LoadTransferFunctionParameters.Resolution; ++KeyIdx)
		{
			float		 Scalar = KeyIdx * KeyIdxToScalar;
			FLinearColor RGBA = TransferFunctionCurve->GetLinearColorValue(Scalar);
			for (int32 Cmpt = 0; Cmpt < 4; ++Cmpt)
			{
				float Value = Cmpt == 0 ? RGBA.R : Cmpt == 1 ? RGBA.G : Cmpt == 2 ? RGBA.B : RGBA.A;
				FlattenDataHalfPrecision.Emplace(Value);
				FlattenDataFullPrecision.Emplace(Value);
			}
		}
	}

	// Setup Texture
	{
		auto TexOrErrMsg = FVolDataTransferFunction::CreateTexture(
			{ .Resolution = LoadTransferFunctionParameters.Resolution, .TFFlatArray = FlattenDataHalfPrecision });
		if (TexOrErrMsg.IsType<FString>())
		{
			UE_LOG(LogVolData, Error, TEXT("%s"), *TexOrErrMsg.Get<FString>());
			return;
		}
		TransferFunction = TexOrErrMsg.Get<UTexture2D*>();
	}

	// Setup Pre-integrated Texture
	{
		auto TexOrErrMsg = FVolDataTransferFunction::CreateTexturePreIntegrated(
			{ .Resolution = LoadTransferFunctionParameters.Resolution, .TFFlatArray = FlattenDataHalfPrecision });
		if (TexOrErrMsg.IsType<FString>())
		{
			UE_LOG(LogVolData, Error, TEXT("%s"), *TexOrErrMsg.Get<FString>());
			return;
		}
		TransferFunctionPreIntegrated = TexOrErrMsg.Get<UTexture2D*>();
	}

	// Save copy of TF in CPU
	CPUData->TransferFunctionData = std::move(FlattenDataFullPrecision);
	CPUData->TransferFunctionDataPreIntegrated = FVolDataTransferFunction::PreIntegrateFromFlatArray<false>(
		CPUData->TransferFunctionData, LoadTransferFunctionParameters.Resolution);

	TransferFunctionChanged.Broadcast(this);
}

void UVolDataVDBComponent::buildVDB(bool bNeedReload, bool bNeedRelayoutAtlas)
{
	bool bNeedFullRebuild = bNeedRelayoutAtlas;
	if (bNeedReload || LoadRAWVolumeParams.bNeedReload)
	{
		loadRAWVolume();

		LoadRAWVolumeParams.bNeedReload = false;
		bNeedFullRebuild = true;
	}

	if (bNeedReload || LoadTransferFunctionParameters.bNeedReload)
	{
		loadTransferFunction();
		syncTransferFunctionFromCurve();

		LoadTransferFunctionParameters.bNeedReload = false;
		bNeedFullRebuild |= LoadTransferFunctionParameters.bNeedFullRebuild;
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

	if (bNeedFullRebuild)
	{
		AsyncTask(ENamedThreads::Type::ActualRenderingThread, [this]() {
			VolData::FStdOutputLinker Linker;
			VDB->FullBuild({ .RAWVolumeData = CPUData->RAWVolumeData.GetData(),
				.EmptyScalarRanges = CPUData->EmptyScalarRanges.GetData(),
				.EmptyScalarRangeNum = CPUData->EmptyScalarRangeNum,
				.MaxAllowedGPUMemoryInGB = VDBParams.MaxAllowedGPUMemoryInGB,
				.MaxAllowedResidentFrameNum = VDBParams.MaxAllowedResidentFrameNum,
				.VDBParams = VDBParams });
		});
	}
}
