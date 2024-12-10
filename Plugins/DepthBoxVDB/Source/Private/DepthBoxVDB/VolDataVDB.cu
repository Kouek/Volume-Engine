#include "VolData.h"

#include <algorithm>
#include <future>

#include <thrust/binary_search.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/system/cuda/execution_policy.h>

#include <thrust/host_vector.h>

#include <glm/gtx/string_cast.hpp>

#include <CUDA/Algorithm.h>

// NDEBUG is set when build type is RelWithDbg, thus we define a new macro to Debug
#define DEPTHBOX_DEBUG

std::shared_ptr<DepthBoxVDB::VolData::IVDB> DepthBoxVDB::VolData::IVDB::Create(
	const CreateParameters& Params)
{
	return std::make_shared<VDB>(Params);
}

DepthBoxVDB::VolData::VDB::VDB(const CreateParameters& Params)
{
	int DeviceNum = 0;
	CUDA_CHECK(cudaGetDeviceCount(&DeviceNum));
	if (DeviceNum == 0)
	{
		std::cerr << "No CUDA Device found.";
	}

	cudaDeviceProp Prop;
	CUDA_CHECK(cudaGetDeviceProperties(&Prop, 0));

	for (uint32_t i = 0; i < static_cast<uint32_t>(EStream::Max); ++i)
	{
		CUDA_CHECK(cudaStreamCreateWithFlags(&Streams[i], cudaStreamNonBlocking));
	}

	invalidate();
}

DepthBoxVDB::VolData::VDB::~VDB()
{
	for (uint32_t i = 0; i < static_cast<uint32_t>(EStream::Max); ++i)
	{
		CUDA_CHECK(cudaStreamDestroy(Streams[i]));
	}
}

#ifdef DEPTHBOX_DEBUG
template <typename T> void DebugInCPU(const thrust::device_vector<T>& dVector, const char* Name)
{
	using namespace DepthBoxVDB::VolData;

	thrust::host_vector<T> hVetcor = dVector;
	std ::string		   DebugMsg = std::format("CUDA Debug {}:\n\t", Name);
	uint32_t			   Index = 0;
	for (auto Itr = hVetcor.begin(); Itr != hVetcor.end(); ++Itr)
	{
		if constexpr (std::is_same_v<T, BrickSortKey>)
		{
			auto& LevPos = Itr.base()->LevelPosition;
			DebugMsg.append(std::format("{}:({},{},{},{}), ", Index,
				static_cast<uint32_t>(LevPos.Level), static_cast<uint32_t>(LevPos.X),
				static_cast<uint32_t>(LevPos.Y), static_cast<uint32_t>(LevPos.Z)));

			if (*Itr.base() == BrickSortKey::Invalid())
				break; // Print valid keys only
		}
		else if constexpr (std::is_same_v<T, glm::vec2>)
		{
			DebugMsg.append(std::format("{}:{}, ", Index, glm::to_string(*Itr.base())));
		}
		else if constexpr (std::is_same_v<T, DepthBoxVDB::VolData::VDBNode>)
		{
			DepthBoxVDB::VolData::VDBNode Node = *Itr.base();
			DebugMsg.append(std::format("{}:({},{},{}), ", Index, glm::to_string(Node.Coord),
				glm::to_string(Node.CoordInAtlas), Node.ChildListOffset));
		}
		else if constexpr (std::is_pod_v<T>)
		{
			DebugMsg.append(std::format("{}:{}, ", Index, *Itr.base()));
		}

		++Index;
	}
	DebugMsg.push_back('\n');

	std::cout << DebugMsg;
}

	#define CUDA_DEBUG_IN_CPU(X) DebugInCPU(X, #X)

#endif

void DepthBoxVDB::VolData::VDB::FullBuild(const FullBuildParameters& Params)
{
	StartAppendFrame({ .EmptyScalarRanges = Params.EmptyScalarRanges,
		.EmptyScalarRangeNum = Params.EmptyScalarRangeNum,
		.MaxAllowedGPUMemoryInGB = Params.MaxAllowedGPUMemoryInGB,
		.MaxAllowedResidentFrameNum = Params.MaxAllowedResidentFrameNum,
		.VDBParams = Params.VDBParams });
	AppendFrame({ .RAWVolumeData = Params.RAWVolumeData });
	EndAppendFrame();
}

void DepthBoxVDB::VolData::VDB::StartAppendFrame(const StartAppendFrameParameters& Params)
{
	invalidate();

	VDBParams = Params.VDBParams;
	MaxAllowedGPUMemoryInByte = static_cast<size_t>(Params.MaxAllowedGPUMemoryInGB) * (1 << 30);
	MaxAllowedResidentFrameNum = Params.MaxAllowedResidentFrameNum;

	EmptyScalarRanges.resize(Params.EmptyScalarRangeNum);
	for (uint32_t RngIdx = 0; RngIdx < Params.EmptyScalarRangeNum; ++RngIdx)
	{
		EmptyScalarRanges[RngIdx] = Params.EmptyScalarRanges[RngIdx];
	}
	EmptyScalarRangesReactive = EmptyScalarRanges;
	dEmptyScalarRanges.resize(EmptyScalarRanges.size());
	CUDA_CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(dEmptyScalarRanges.data()),
		EmptyScalarRanges.data(), sizeof(glm::vec2) * EmptyScalarRanges.size(),
		cudaMemcpyHostToDevice, getStream(EStream::Copy)));
}

void DepthBoxVDB::VolData::VDB::AppendFrame(const AppendFrameParameters& Params)
{
	uint32_t FrameIndex = GetFrameNum();
	generateDataPerFrame(Params.RAWVolumeData, FrameIndex);
}

void DepthBoxVDB::VolData::VDB::EndAppendFrame()
{
	if (!allocateResource())
		return;

	SwitchToFrame(0);
}

void DepthBoxVDB::VolData::VDB::Popup(void* Params)
{
	auto PopupParams = *(DepthBoxVDB::VolData::VDB::PopupFrameParameters*)Params;
	PopupParams.OutVDB->dVDBDataCurrentFrame = PopupParams.InRsdDataPF->dVDBData;
}

void DepthBoxVDB::VolData::VDB::SwitchToFrame(uint32_t FrameIndex)
{
	waitForAllStream();
	switchToFrame(FrameIndex);
}

void DepthBoxVDB::VolData::VDB::UpdateDepthBox(const UpdateDepthBoxParameters& Params)
{
	EmptyScalarRangesReactive.resize(Params.EmptyScalarRangeNum);
	for (uint32_t RngIdx = 0; RngIdx < Params.EmptyScalarRangeNum; ++RngIdx)
	{
		EmptyScalarRangesReactive[RngIdx] = Params.EmptyScalarRanges[RngIdx];
	}
	CUDA_CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(dEmptyScalarRanges.data()),
		EmptyScalarRangesReactive.data(), sizeof(glm::vec2) * EmptyScalarRangesReactive.size(),
		cudaMemcpyHostToDevice, getStream(EStream::Copy)));

	for (uint32_t FrameIndex = 0; FrameIndex < GetFrameNum(); ++FrameIndex)
	{
		DataPerFrames[FrameIndex].bUpdatedFromEmptyScalarRanges = false;
	}
}

void DepthBoxVDB::VolData::VDB::generateDataPerFrame(
	const uint8_t* RAWVolumeData, uint32_t FrameIndex)
{
	uint32_t VoxelYxX = VDBParams.VoxelPerVolume.x * VDBParams.VoxelPerVolume.y;
	uint32_t VoxelYxXPerAtlasBrick = VDBParams.VoxelPerAtlasBrick * VDBParams.VoxelPerAtlasBrick;
	uint32_t BrickYxX = VDBParams.BrickPerVolume.x * VDBParams.BrickPerVolume.y;
	uint32_t BrickNum = BrickYxX * VDBParams.BrickPerVolume.z;
	uint32_t MaxCoordValInAtlasBrick =
		VDBParams.ApronAndDepthWidth + VDBParams.ChildPerLevels[0] - 1;
	uint32_t VoxelNumPerAtlasBrick = VoxelYxXPerAtlasBrick * VDBParams.VoxelPerAtlasBrick;
	uint32_t VoxelApronOffset = VDBParams.ApronAndDepthWidth - VDBParams.ApronWidth;
	uint32_t VoxelNumPerBrick = static_cast<uint32_t>(VDBParams.ChildPerLevels[0]);
	VoxelNumPerBrick = VoxelNumPerBrick * VoxelNumPerBrick * VoxelNumPerBrick;

	std::vector<std::future<void>> Futures(BrickNum);

	if (DataPerFrames.size() <= FrameIndex)
	{
		DataPerFrames.resize(FrameIndex + 1);
	}
	auto& BrickedData = DataPerFrames[FrameIndex].BrickedData;
	BrickedData.resize(SizeOfVoxelType(VDBParams.VoxelType) * BrickNum * VoxelNumPerAtlasBrick);
	auto& BrickSortKeys = DataPerFrames[FrameIndex].BrickSortKeys;
	BrickSortKeys.clear();
	auto& dBrickSortKeys = DataPerFrames[FrameIndex].dBrickSortKeys;

	std::vector<uint8_t> BrickValids(BrickNum, 0);
	auto Assign = [&]<typename T>(T* Dst, const T* Src, const CoordType& BrickCoord) {
		auto Sample = [&](CoordType Coord) -> T {
			Coord = glm::clamp(Coord, CoordType(0), VDBParams.VoxelPerVolume - 1);
			return Src[Coord.z * VoxelYxX + Coord.y * VDBParams.VoxelPerVolume.x + Coord.x];
		};

		CoordType MinCoord = BrickCoord * VDBParams.ChildPerLevels[0];
		uint32_t  BrickIndex =
			BrickCoord.z * BrickYxX + BrickCoord.y * VDBParams.BrickPerVolume.x + BrickCoord.x;

		uint32_t  EmptyVoxelNum = 0;
		T*		  DstPitchPtr = nullptr;
		CoordType dCoord;
		for (dCoord.z = VoxelApronOffset;
			 dCoord.z < VDBParams.VoxelPerAtlasBrick - VoxelApronOffset; ++dCoord.z)
		{
			DstPitchPtr = Dst + BrickIndex * VoxelNumPerAtlasBrick
				+ dCoord.z * VoxelYxXPerAtlasBrick
				+ VoxelApronOffset * VDBParams.VoxelPerAtlasBrick;

			for (dCoord.y = VoxelApronOffset;
				 dCoord.y < VDBParams.VoxelPerAtlasBrick - VoxelApronOffset; ++dCoord.y)
			{
				for (dCoord.x = VoxelApronOffset;
					 dCoord.x < VDBParams.VoxelPerAtlasBrick - VoxelApronOffset; ++dCoord.x)
				{
					T Scalar = Sample(MinCoord + dCoord - VDBParams.ApronAndDepthWidth);
					DstPitchPtr[dCoord.x] = Scalar;

					bool bInBrick = true;
					for (int32_t Axis = 0; Axis < 3; ++Axis)
					{
						if (dCoord[Axis] < VDBParams.ApronAndDepthWidth
							|| dCoord[Axis] > MaxCoordValInAtlasBrick)
						{
							bInBrick = false;
							break;
						}
					}
					if (!bInBrick)
						continue;

					for (uint32_t RngIdx = 0; RngIdx < EmptyScalarRanges.size(); ++RngIdx)
					{
						glm::vec2 Range = EmptyScalarRanges[RngIdx];
						if (Range[0] <= Scalar && Scalar <= Range[1])
						{
							++EmptyVoxelNum;
							break;
						}
					}
				}

				DstPitchPtr += VDBParams.VoxelPerAtlasBrick;
			}
		}

		if (EmptyVoxelNum < VoxelNumPerBrick)
		{
			BrickValids[BrickIndex] = 1;
		}
	};

	{
		uint32_t  BrickIndex = 0;
		CoordType BrickCoord;
		for (BrickCoord.z = 0; BrickCoord.z < VDBParams.BrickPerVolume.z; ++BrickCoord.z)
			for (BrickCoord.y = 0; BrickCoord.y < VDBParams.BrickPerVolume.y; ++BrickCoord.y)
				for (BrickCoord.x = 0; BrickCoord.x < VDBParams.BrickPerVolume.x; ++BrickCoord.x)
				{
					switch (VDBParams.VoxelType)
					{
						case EVoxelType::UInt8:
							Futures[BrickIndex] =
								std::async(Assign, reinterpret_cast<uint8_t*>(BrickedData.data()),
									reinterpret_cast<const uint8_t*>(RAWVolumeData), BrickCoord);
							break;
						case EVoxelType::UInt16:
							Futures[BrickIndex] =
								std::async(Assign, reinterpret_cast<uint16_t*>(BrickedData.data()),
									reinterpret_cast<const uint16_t*>(RAWVolumeData), BrickCoord);
							break;
						case EVoxelType::Float32:
							Futures[BrickIndex] =
								std::async(Assign, reinterpret_cast<float*>(BrickedData.data()),
									reinterpret_cast<const float*>(RAWVolumeData), BrickCoord);
							break;
						default:
							assert(false);
					}

					++BrickIndex;
				}
		for (auto& Future : Futures)
		{
			Future.wait();
		}
	}

	{
		uint32_t ValidBrickNum = std::count_if(
			BrickValids.begin(), BrickValids.end(), [](uint8_t Valid) { return Valid == 1; });

		BrickSortKeys.reserve(ValidBrickNum);
		BrickSortKey BSKey;
		BSKey.LevelPosition.Level = 0;
		for (uint32_t BrickIndex = 0; BrickIndex < BrickNum; ++BrickIndex)
		{
			if (BrickValids[BrickIndex] == 0)
				continue;

			uint32_t		   BrickIndexWithFrame = FrameIndex * BrickNum + BrickIndex;
			CoordWithFrameType BrickCoordWithFrame = BrickIndexToCoord(BrickIndexWithFrame);

			BSKey.LevelPosition.X = BrickCoordWithFrame.x;
			BSKey.LevelPosition.Y = BrickCoordWithFrame.y;
			BSKey.LevelPosition.Z = BrickCoordWithFrame.z;
			BrickSortKeys.emplace_back(BSKey);
		}

		dBrickSortKeys.resize(ValidBrickNum);
		CUDA_CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(dBrickSortKeys.data()),
			BrickSortKeys.data(), sizeof(BrickSortKey) * ValidBrickNum, cudaMemcpyHostToDevice,
			getStream(EStream::Copy)));
	}
}

bool DepthBoxVDB::VolData::VDB::allocateResource()
{
	uint32_t FrameNum = GetFrameNum();

	// Compute NeededVoxelPerAtlas
	CoordType NeededVoxelPerAtlas;
	{
		uint32_t NeededValidBrickNumInPlayLoop = 0;
		for (MaxResidentFrameNum = 1;
			 MaxResidentFrameNum <= std::min(FrameNum, MaxAllowedResidentFrameNum);
			 ++MaxResidentFrameNum)
		{
			size_t NeededGPUMemInByteInPlayLoop = 0;
			// TODO: To avoid double for-loop, rmeove head and add tail
			for (uint32_t StartFrmIdx = 0; StartFrmIdx < FrameNum; ++StartFrmIdx)
			{
				uint32_t EndFrmIdx = StartFrmIdx + MaxResidentFrameNum - 1;
				size_t	 NeededGPUMemInByte = 0;
				for (uint32_t FrameIndex = StartFrmIdx; FrameIndex <= EndFrmIdx; ++FrameIndex)
				{
					NeededGPUMemInByte += DataPerFrames[FrameIndex % FrameNum].BrickSortKeys.size();
				}

				NeededValidBrickNumInPlayLoop = std::max(
					NeededValidBrickNumInPlayLoop, static_cast<uint32_t>(NeededGPUMemInByte));
				NeededGPUMemInByte *= SizeOfVoxelType(VDBParams.VoxelType)
					* VDBParams.VoxelPerAtlasBrick * VDBParams.VoxelPerAtlasBrick
					* VDBParams.VoxelPerAtlasBrick;
				NeededGPUMemInByteInPlayLoop =
					std::max(NeededGPUMemInByteInPlayLoop, NeededGPUMemInByte);
			}

			if (NeededGPUMemInByteInPlayLoop > MaxAllowedGPUMemoryInByte)
				break;
		}
		--MaxResidentFrameNum;

		if (MaxResidentFrameNum == 0)
		{
			std::cerr << "MaxAllowedGPUMemory is too small.\n";
			return false;
		}

		NeededVoxelPerAtlas.x = VDBParams.BrickPerVolume.x;
		NeededVoxelPerAtlas.y = VDBParams.BrickPerVolume.y;
		NeededVoxelPerAtlas.z = 1;
		while ([&]() {
			return static_cast<uint32_t>(NeededVoxelPerAtlas.z) * NeededVoxelPerAtlas.y
				* NeededVoxelPerAtlas.x
				< NeededValidBrickNumInPlayLoop;
		}())
		{
			++NeededVoxelPerAtlas.z;
		}
		BrickPerAtlas = NeededVoxelPerAtlas;
		NeededVoxelPerAtlas *= VDBParams.VoxelPerAtlasBrick;

		size_t AtlasGPUMemInByte = SizeOfVoxelType(VDBParams.VoxelType) * NeededVoxelPerAtlas.z
			* NeededVoxelPerAtlas.y * NeededVoxelPerAtlas.z;
		if (AtlasGPUMemInByte > MaxAllowedGPUMemoryInByte)
		{
			std::cerr << "MaxAllowedGPUMemory is too small.\n";
			return false;
		}
	}

	// Resize Atlas
	{
		cudaChannelFormatDesc ChannelDesc;
		switch (VDBParams.VoxelType)
		{
			case EVoxelType::UInt8:
				ChannelDesc = cudaCreateChannelDesc<uint8_t>();
				break;
			case EVoxelType::UInt16:
				ChannelDesc = cudaCreateChannelDesc<uint16_t>();
				break;
			case EVoxelType::Float32:
				ChannelDesc = cudaCreateChannelDesc<float>();
				break;
			default:
				assert(false);
		}
		AtlasArray = std::make_shared<CUDA::Array>(ChannelDesc, NeededVoxelPerAtlas);
	}

	// Create Texture and Surface
	AtlasTexture = std::make_unique<CUDA::Texture>(AtlasArray);
	AtlasSurface = std::make_unique<CUDA::Surface>(AtlasArray);

	// Init Mapping Tables
	BrickWithFrameToAtlasBrick.assign(FrameNum * VDBParams.BrickPerVolume.z
			* VDBParams.BrickPerVolume.y * VDBParams.BrickPerVolume.z,
		kInvalidIndex);
	dBrickWithFrameToAtlasBrick.resize(BrickWithFrameToAtlasBrick.size());

	uint32_t AtlasBrickNum =
		static_cast<uint32_t>(BrickPerAtlas.z) * BrickPerAtlas.y * BrickPerAtlas.x;
	AtlasBrickToBrickWithFrame.assign(AtlasBrickNum, kInvalidIndex);
	dAtlasBrickToBrickWithFrame.resize(AtlasBrickNum);

	AvailableAtlasBrick.reserve(AtlasBrickNum);
	for (uint32_t BrickIndex = 0; BrickIndex < AtlasBrickNum; ++BrickIndex)
	{
		AvailableAtlasBrick.emplace_back(BrickIndex);
	}

	// Resize Resident Frames
	ResidentDataPerFrames.clear();
	ResidentDataPerFrames.resize(MaxResidentFrameNum);

	return true;
}

void DepthBoxVDB::VolData::VDB::transferBrickDataToAtlas(uint32_t ResidentIndex)
{
	uint32_t VoxelNumPerAtlasBrick = static_cast<uint32_t>(VDBParams.VoxelPerAtlasBrick)
		* VDBParams.VoxelPerAtlasBrick * VDBParams.VoxelPerAtlasBrick;
	uint32_t AtlasBrickNum =
		static_cast<uint32_t>(BrickPerAtlas.z) * BrickPerAtlas.y * BrickPerAtlas.x;
	uint32_t BrickNum = static_cast<uint32_t>(VDBParams.BrickPerVolume.z)
		* VDBParams.BrickPerVolume.y * VDBParams.BrickPerVolume.x;

	auto&	 RsdDataPF = ResidentDataPerFrames[ResidentIndex];
	uint32_t FrameIndex = RsdDataPF.FrameIndex;
	auto&	 BrickedData = DataPerFrames[FrameIndex].BrickedData;

	auto TransferBrick = [&]<typename T>(T* Src, uint32_t BrickIndex, uint32_t AtlasBrickIndex) {
		CoordType AtlasBrickCoord = AtlasBrickIndexToCoord(AtlasBrickIndex);

		cudaMemcpy3DParms MemCpyParams{};
		MemCpyParams.srcPtr = make_cudaPitchedPtr(Src + BrickIndex * VoxelNumPerAtlasBrick,
			sizeof(T) * VDBParams.VoxelPerAtlasBrick, VDBParams.VoxelPerAtlasBrick,
			VDBParams.VoxelPerAtlasBrick);
		MemCpyParams.extent = make_cudaExtent(VDBParams.VoxelPerAtlasBrick,
			VDBParams.VoxelPerAtlasBrick, VDBParams.VoxelPerAtlasBrick);
		MemCpyParams.dstArray = AtlasArray->Get();
		MemCpyParams.dstPos.x = AtlasBrickCoord.x * VDBParams.VoxelPerAtlasBrick;
		MemCpyParams.dstPos.y = AtlasBrickCoord.y * VDBParams.VoxelPerAtlasBrick;
		MemCpyParams.dstPos.z = AtlasBrickCoord.z * VDBParams.VoxelPerAtlasBrick;
		MemCpyParams.kind = cudaMemcpyHostToDevice;

		CUDA_CHECK(cudaMemcpy3DAsync(&MemCpyParams, getStream(EStream::Copy)));
	};

	RsdDataPF.Wait(getStream(EStream::Copy), ResidentDataPerFrame::EEvent::UpdateDepthBox);
	RsdDataPF.Wait(getStream(EStream::Copy), ResidentDataPerFrame::EEvent::BuildVDB);
	for (auto [BrickIndexWithFrame, AtlasBrickIndex] : RsdDataPF.BrickWithFrameToAtlasBrick)
	{
		uint32_t BrickIndex = BrickIndexWithFrame - FrameIndex * BrickNum;
		switch (VDBParams.VoxelType)
		{
			case EVoxelType::UInt8:
				TransferBrick(
					reinterpret_cast<uint8_t*>(BrickedData.data()), BrickIndex, AtlasBrickIndex);
				break;
			case EVoxelType::UInt16:
				TransferBrick(
					reinterpret_cast<uint16_t*>(BrickedData.data()), BrickIndex, AtlasBrickIndex);
				break;
			case EVoxelType::Float32:
				TransferBrick(
					reinterpret_cast<float*>(BrickedData.data()), BrickIndex, AtlasBrickIndex);
				break;
			default:
				assert(false);
		}
	}
	// Transfer Mapping
	{
		uint32_t* dPtr = thrust::raw_pointer_cast(dBrickWithFrameToAtlasBrick.data());
		dPtr += FrameIndex * BrickNum;
		uint32_t* Ptr = BrickWithFrameToAtlasBrick.data();
		Ptr += FrameIndex * BrickNum;
		CUDA_CHECK(cudaMemcpyAsync(dPtr, Ptr, sizeof(uint32_t) * BrickNum, cudaMemcpyHostToDevice,
			getStream(EStream::Copy)));

		// Inverse mapping cannot be copied incrementally
		dPtr = thrust::raw_pointer_cast(dAtlasBrickToBrickWithFrame.data());
		Ptr = AtlasBrickToBrickWithFrame.data();
		CUDA_CHECK(cudaMemcpyAsync(dPtr, Ptr, sizeof(uint32_t) * AtlasBrickNum,
			cudaMemcpyHostToDevice, getStream(EStream::Copy)));
	}
	RsdDataPF.Record(
		ResidentDataPerFrame::EEvent::TransferBrickDataToAtlas, getStream(EStream::Copy));
}

void DepthBoxVDB::VolData::VDB::updateDepthBox(uint32_t ResidentIndex)
{
	auto&	 RsdDataPF = ResidentDataPerFrames[ResidentIndex];
	uint32_t FrameIndex = RsdDataPF.FrameIndex;
	auto&	 DataPF = DataPerFrames[FrameIndex];
	if (DataPF.bUpdatedFromEmptyScalarRanges)
		return;

	RsdDataPF.Wait(
		getStream(EStream::Atlas), ResidentDataPerFrame::EEvent::TransferBrickDataToAtlas);
	switch (VDBParams.VoxelType)
	{
		case EVoxelType::UInt8:
			updateDepthBox<uint8_t>(FrameIndex);
			break;
		case EVoxelType::UInt16:
			updateDepthBox<uint16_t>(FrameIndex);
			break;
		case EVoxelType::Float32:
			updateDepthBox<float>(FrameIndex);
			break;
		default:
			assert(false);
	}
	RsdDataPF.Record(ResidentDataPerFrame::EEvent::UpdateDepthBox, getStream(EStream::Atlas));

	DataPF.bUpdatedFromEmptyScalarRanges = true;
	DataPF.bTransferredToCPU = false;
}

template <typename VoxelType> void DepthBoxVDB::VolData::VDB::updateDepthBox(uint32_t FrameIndex)
{
	uint32_t BrickYxXPerAtlas = BrickPerAtlas.x * BrickPerAtlas.y;
	uint32_t VoxelYxXPerBrick = VDBParams.ChildPerLevels[0] * VDBParams.ChildPerLevels[0];
	uint32_t BrickNumPerAtlas =
		static_cast<uint32_t>(BrickPerAtlas.z) * BrickPerAtlas.y * BrickPerAtlas.x;
	uint32_t DepthVoxelNumPerAtlas = BrickNumPerAtlas * 6 * VoxelYxXPerBrick;

	auto UpdateKernel = [DepthVoxelNumPerAtlas, VoxelYxXPerBrick, BrickYxXPerAtlas,
							BrickPerAtlas = BrickPerAtlas,
							EmptyScalarRangeNum = static_cast<uint32_t>(EmptyScalarRanges.size()),
							EmptyScalarRanges = thrust::raw_pointer_cast(dEmptyScalarRanges.data()),
							AtlasSurface = AtlasSurface->Get(), AtlasTexture = AtlasTexture->Get(),
							VDBParams = VDBParams] __device__(uint32_t DepthVoxelIndex) {
		uint32_t  AtlasBrickIndex = DepthVoxelIndex / (6 * VoxelYxXPerBrick);
		CoordType AtlasBrickCoord;
		AtlasBrickCoord.z = AtlasBrickIndex / BrickYxXPerAtlas;
		uint32_t Tmp = AtlasBrickIndex - AtlasBrickCoord.z * BrickYxXPerAtlas;
		AtlasBrickCoord.y = Tmp / BrickPerAtlas.x;
		AtlasBrickCoord.x = Tmp - AtlasBrickCoord.y * BrickPerAtlas.x;

		Tmp = DepthVoxelIndex - AtlasBrickIndex * (6 * VoxelYxXPerBrick);
		uint32_t FaceIndex = Tmp / VoxelYxXPerBrick;
		Tmp = Tmp - FaceIndex * VoxelYxXPerBrick;

		// In Brick
		CoordType Coord(0);
		Coord.x = Tmp / VDBParams.ChildPerLevels[0];
		Coord.y = Tmp - Coord.x * VDBParams.ChildPerLevels[0];
		switch (FaceIndex)
		{
			case 2:
			case 3:
				Coord.z = Coord.x;
				break;
			case 4:
			case 5:
				Coord.z = Coord.y;
				break;
		}
		switch (FaceIndex)
		{
			case 0:
				Coord.z = 0;
				break;
			case 1:
				Coord.z = VDBParams.ChildPerLevels[0] - 1;
				break;
			case 2:
				Coord.x = 0;
				break;
			case 3:
				Coord.x = VDBParams.ChildPerLevels[0] - 1;
				break;
			case 4:
				Coord.y = 0;
				break;
			case 5:
				Coord.y = VDBParams.ChildPerLevels[0] - 1;
				break;
		}

		CoordType DepthCoord = Coord;
		switch (FaceIndex)
		{
			case 0:
			case 1:
				DepthCoord.z = VDBParams.DepthCoordValueInAtlasBrick[FaceIndex % 2];
				break;
			case 2:
			case 3:
				DepthCoord.x = VDBParams.DepthCoordValueInAtlasBrick[FaceIndex % 2];
				break;
			case 4:
			case 5:
				DepthCoord.y = VDBParams.DepthCoordValueInAtlasBrick[FaceIndex % 2];
				break;
		}

		// In Brick to In Atlas
		Coord =
			AtlasBrickCoord * VDBParams.VoxelPerAtlasBrick + VDBParams.ApronAndDepthWidth + Coord;
		DepthCoord = AtlasBrickCoord * VDBParams.VoxelPerAtlasBrick + VDBParams.ApronAndDepthWidth
			+ DepthCoord;

		CoordType Rht(0);
		switch (FaceIndex)
		{
			case 0:
			case 1:
			case 4:
			case 5:
				Rht.x = 1;
				break;
			case 2:
			case 3:
				Rht.z = 1;
				break;
		}
		CoordType Up(0);
		switch (FaceIndex)
		{
			case 0:
			case 1:
			case 2:
			case 3:
				Up.y = 1;
				break;
			case 4:
			case 5:
				Up.z = 1;
				break;
		}

		auto IsEmpty = [&](const CoordType& Coord) {
			VoxelType Scalar =
				surf3Dread<VoxelType>(AtlasSurface, sizeof(VoxelType) * Coord.x, Coord.y, Coord.z);
			for (uint32_t RngIdx = 0; RngIdx < EmptyScalarRangeNum; ++RngIdx)
			{
				glm::vec2 Range = EmptyScalarRanges[RngIdx];
				if (Range[0] <= Scalar && Scalar <= Range[1])
				{
					return true;
				}
			}
			return false;
		};
		VoxelType Depth = 0;
		while (true)
		{
			bool bEmpty = true;
#ifdef __CUDA_ARCH__
	#pragma unroll
#endif
			for (CoordValueType RhtVal = -1; RhtVal <= 1; ++RhtVal)
#ifdef __CUDA_ARCH__
	#pragma unroll
#endif
				for (CoordValueType UpVal = -1; UpVal <= 1; ++UpVal)
				{
					bEmpty &= IsEmpty(Coord + RhtVal * Rht + UpVal * Up);
				}
			if (!bEmpty || Depth >= VDBParams.ChildPerLevels[0] - 1)
				break;

			switch (FaceIndex)
			{
				case 0:
					Coord.z += 1;
					break;
				case 1:
					Coord.z -= 1;
					break;
				case 2:
					Coord.x += 1;
					break;
				case 3:
					Coord.x -= 1;
					break;
				case 4:
					Coord.y += 1;
					break;
				case 5:
					Coord.y -= 1;
					break;
			}
			++Depth;
		}
		Depth = Depth == 1 ? 0 : Depth;

		// Debug FaceIndex
		// Depth = FaceIndex;
		surf3Dwrite(
			Depth, AtlasSurface, sizeof(VoxelType) * DepthCoord.x, DepthCoord.y, DepthCoord.z);
	};

	thrust::for_each(thrust::cuda::par_nosync.on(getStream(EStream::Atlas)),
		thrust::make_counting_iterator(uint32_t(0)),
		thrust::make_counting_iterator(DepthVoxelNumPerAtlas), UpdateKernel);
}
template void DepthBoxVDB::VolData::VDB::updateDepthBox<uint8_t>(uint32_t FrameIndex);
template void DepthBoxVDB::VolData::VDB::updateDepthBox<uint16_t>(uint32_t FrameIndex);
template void DepthBoxVDB::VolData::VDB::updateDepthBox<float>(uint32_t FrameIndex);

void DepthBoxVDB::VolData::VDB::transferBrickDataToCPU(uint32_t ResidentIndex)
{
	auto&	 RsdDataPF = ResidentDataPerFrames[ResidentIndex];
	uint32_t FrameIndex = RsdDataPF.FrameIndex;
	auto&	 DataPF = DataPerFrames[FrameIndex];
	if (!DataPF.bTransferredToCPU)
		return;

	uint32_t BrickNum = static_cast<uint32_t>(VDBParams.BrickPerVolume.z)
		* VDBParams.BrickPerVolume.y * VDBParams.BrickPerVolume.x;
	uint32_t VoxelNumPerAtlasBrick = static_cast<uint32_t>(VDBParams.VoxelPerAtlasBrick)
		* VDBParams.VoxelPerAtlasBrick * VDBParams.VoxelPerAtlasBrick;

	auto& BrickedData = DataPerFrames[FrameIndex].BrickedData;

	auto Transfer = [&]<typename T>(T* Dst, uint32_t BrickIndex, uint32_t AtlasBrickIndex) {
		CoordType AtlasBrickCoord = AtlasBrickIndexToCoord(AtlasBrickIndex);

		cudaMemcpy3DParms MemCpyParams{};
		MemCpyParams.srcArray = AtlasArray->Get();
		MemCpyParams.srcPos.x = AtlasBrickCoord.x * VDBParams.VoxelPerAtlasBrick;
		MemCpyParams.srcPos.y = AtlasBrickCoord.y * VDBParams.VoxelPerAtlasBrick;
		MemCpyParams.srcPos.z = AtlasBrickCoord.z * VDBParams.VoxelPerAtlasBrick;
		MemCpyParams.extent = make_cudaExtent(VDBParams.VoxelPerAtlasBrick,
			VDBParams.VoxelPerAtlasBrick, VDBParams.VoxelPerAtlasBrick);
		MemCpyParams.dstPtr = make_cudaPitchedPtr(Dst + BrickIndex * VoxelNumPerAtlasBrick,
			sizeof(T) * VDBParams.VoxelPerAtlasBrick, VDBParams.VoxelPerAtlasBrick,
			VDBParams.VoxelPerAtlasBrick);
		MemCpyParams.kind = cudaMemcpyDeviceToHost;

		CUDA_CHECK(cudaMemcpy3DAsync(&MemCpyParams, getStream(EStream::Copy)));
	};

	RsdDataPF.Wait(getStream(EStream::Copy), ResidentDataPerFrame::EEvent::UpdateDepthBox);
	for (auto [BrickIndexWithFrame, AtlasBrickIndex] : RsdDataPF.BrickWithFrameToAtlasBrick)
	{
		uint32_t BrickIndex = BrickIndexWithFrame - FrameIndex * BrickNum;
		switch (VDBParams.VoxelType)
		{
			case EVoxelType::UInt8:
				Transfer(
					reinterpret_cast<uint8_t*>(BrickedData.data()), BrickIndex, AtlasBrickIndex);
				break;
			case EVoxelType::UInt16:
				Transfer(
					reinterpret_cast<uint16_t*>(BrickedData.data()), BrickIndex, AtlasBrickIndex);
				break;
			case EVoxelType::Float32:
				Transfer(reinterpret_cast<float*>(BrickedData.data()), BrickIndex, AtlasBrickIndex);
				break;
			default:
				assert(false);
		}
	}
	RsdDataPF.Record(
		ResidentDataPerFrame::EEvent::TransferBrickDataToCPU, getStream(EStream::Copy));

	DataPF.bTransferredToCPU = false;
}

void DepthBoxVDB::VolData::VDB::buildVDB(uint32_t ResidentIndex)
{
	auto& RsdDataPF = ResidentDataPerFrames[ResidentIndex];
	if (RsdDataPF.BrickWithFrameToAtlasBrick.empty())
		return;

	auto& dNodePerLevels = RsdDataPF.dNodePerLevels;
	auto& dChildPerLevels = RsdDataPF.dChildPerLevels;
	auto& dVDBData = RsdDataPF.dVDBData;

	uint32_t FrameIndex = RsdDataPF.FrameIndex;
	uint32_t ValidBrickNum = DataPerFrames[FrameIndex].BrickSortKeys.size();
	uint32_t BrickYxX = VDBParams.BrickPerVolume.x * VDBParams.BrickPerVolume.y;
	uint32_t BrickNum = VDBParams.BrickPerVolume.z * BrickYxX;

	auto& DataPF = DataPerFrames[FrameIndex];

	RsdDataPF.Wait(getStream(EStream::VDB), ResidentDataPerFrame::EEvent::TransferBrickDataToAtlas);
	RsdDataPF.Wait(getStream(EStream::VDB), ResidentDataPerFrame::EEvent::UpdateDepthBox);

	// Assign Brick Sort Keys to non-emptY Brick at level 0
	thrust::device_vector<BrickSortKey> dBrickSortKeys(
		(VDBParams.RootLevel + 1) * ValidBrickNum, BrickSortKey::Invalid());
	CUDA_CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(dBrickSortKeys.data()),
		thrust::raw_pointer_cast(DataPF.dBrickSortKeys.data()),
		sizeof(BrickSortKey) * ValidBrickNum, cudaMemcpyDeviceToDevice, getStream(EStream::VDB)));

	// 1. Assign Brick Sort Keys to non-empty Brick at level 1,2,...
	// 2. Sort Brick Sort Keys
	{
		auto AssignBrickKeysKernel = [BrickSortKeys =
											 thrust::raw_pointer_cast(dBrickSortKeys.data()),
										 BrickYxX, ValidBrickNum = ValidBrickNum,
										 VDBParams = VDBParams] __device__(uint32_t ValidIndex) {
			BrickSortKey BSKey = BrickSortKeys[ValidIndex];
			uint64_t	 BSKIndex = ValidIndex;
			for (int32_t Lev = 1; Lev <= VDBParams.RootLevel; ++Lev)
			{
				BSKey.LevelPosition.Level = Lev;
				BSKey.LevelPosition.X /= VDBParams.ChildPerLevels[Lev];
				BSKey.LevelPosition.Y /= VDBParams.ChildPerLevels[Lev];
				BSKey.LevelPosition.Z /= VDBParams.ChildPerLevels[Lev];

				BSKIndex = Lev * ValidBrickNum + ValidIndex;
				BrickSortKeys[BSKIndex] = BSKey;
			}
		};

		thrust::for_each(thrust::cuda::par_nosync.on(getStream(EStream::VDB)),
			thrust::make_counting_iterator(uint32_t(0)),
			thrust::make_counting_iterator(ValidBrickNum), AssignBrickKeysKernel);

		thrust::sort(thrust::cuda::par_nosync.on(getStream(EStream::VDB)), dBrickSortKeys.begin(),
			dBrickSortKeys.end());
	}

	// Compact Brick Sort Keys
	{
		auto dDiffs = CUDA::Difference<uint32_t>(dBrickSortKeys, 0, getStream(EStream::VDB));
		dBrickSortKeys =
			CUDA::Compact(dBrickSortKeys, dDiffs, uint32_t(0), getStream(EStream::VDB));

#ifdef DEPTHBOX_DEBUG
		CUDA_CHECK(cudaStreamSynchronize(getStream(EStream::VDB)));
		CUDA_DEBUG_IN_CPU(dBrickSortKeys);
#endif
	}

	// Allocate Node and Child Pools
	{
		uint32_t StartCurrLev = 0;
		for (int32_t Lev = 0; Lev <= VDBParams.RootLevel; ++Lev)
		{
			uint32_t NumCurrLev = Lev == 0 ? ValidBrickNum
				: Lev == VDBParams.RootLevel
				? 1
				: [&]() {
					  BrickSortKey KeyNextLev;
					  KeyNextLev.LevelPosition.Level = Lev + 1;
					  KeyNextLev.LevelPosition.X = KeyNextLev.LevelPosition.Y =
						  KeyNextLev.LevelPosition.Z = 0;

					  auto ItrCurrLev = dBrickSortKeys.begin() + StartCurrLev;
					  auto ItrNextLev =
						  thrust::lower_bound(thrust::cuda::par.on(getStream(EStream::VDB)),
							  ItrCurrLev, dBrickSortKeys.end(), KeyNextLev);

					  return thrust::distance(ItrCurrLev, ItrNextLev);
				  }();
			StartCurrLev += NumCurrLev;

			dNodePerLevels[Lev].assign(NumCurrLev, VDBNode::CreateInvalid());

			if (Lev > 0)
			{
				uint64_t ChildCurrLev = VDBParams.ChildPerLevels[Lev];
				dChildPerLevels[Lev - 1].assign(
					dNodePerLevels[Lev].size() * ChildCurrLev * ChildCurrLev * ChildCurrLev,
					VDBData::kInvalidChild);
			}
		}
	}

	// Upload
	VDBData VDBData;
	VDBData.VDBParams = VDBParams;
	VDBData.AtlasTexture = AtlasTexture->Get();
	VDBData.AtlasSurface = AtlasSurface->Get();
	for (int32_t Lev = 0; Lev <= VDBParams.RootLevel; ++Lev)
	{
		VDBData.NodePerLevels[Lev] = thrust::raw_pointer_cast(dNodePerLevels[Lev].data());
		if (Lev > 0)
		{
			VDBData.ChildPerLevels[Lev - 1] =
				thrust::raw_pointer_cast(dChildPerLevels[Lev - 1].data());
		}
	}
	CUDA_CHECK(cudaMemcpyAsync(
		dVDBData, &VDBData, sizeof(VDBData), cudaMemcpyHostToDevice, getStream(EStream::VDB)));

	// Assign Node and Child Pools
	{
		auto AssignNodePoolsKernel = [BrickYxX, BrickNum, FrameIndex,
										 BrickPerAtlas = this->BrickPerAtlas, VDBData = dVDBData,
										 BrickWithFrameToAtlasBrick = thrust::raw_pointer_cast(
											 dBrickWithFrameToAtlasBrick.data()),
										 BrickSortKeys = thrust::raw_pointer_cast(
											 dBrickSortKeys.data())] __device__(int32_t Level,
										 uint64_t NodeIndexStart, uint32_t NodeIndex) {
			auto& VDBParams = VDBData->VDBParams;

			VDBNode		 Node;
			BrickSortKey BSKey = BrickSortKeys[NodeIndexStart + NodeIndex];
			Node.Coord =
				CoordType(BSKey.LevelPosition.X, BSKey.LevelPosition.Y, BSKey.LevelPosition.Z);

			if (Level == 0)
			{
				uint32_t BrickIndexWithFrame = FrameIndex * BrickNum + Node.Coord.z * BrickYxX
					+ Node.Coord.y * VDBParams.BrickPerVolume.x + Node.Coord.x;
				Node.CoordInAtlas =
					IndexToCoord(BrickWithFrameToAtlasBrick[BrickIndexWithFrame], BrickPerAtlas);
			}
			else
			{
				int32_t ChildCurrLev = VDBParams.ChildPerLevels[Level];
				Node.ChildListOffset =
					static_cast<uint64_t>(NodeIndex) * ChildCurrLev * ChildCurrLev * ChildCurrLev;
			}

			VDBData->Node(Level, NodeIndex) = Node;
		};

		uint64_t NodeIndexStart = 0;
		for (int32_t Lev = 0; Lev < VDBParams.RootLevel; ++Lev)
		{
			thrust::for_each(thrust::cuda::par_nosync.on(getStream(EStream::VDB)),
				thrust::make_counting_iterator(uint32_t(0)),
				thrust::make_counting_iterator(static_cast<uint32_t>(dNodePerLevels[Lev].size())),
				[Lev, NodeIndexStart, AssignNodePoolsKernel] __device__(
					uint32_t NodeIndex) { AssignNodePoolsKernel(Lev, NodeIndexStart, NodeIndex); });
			NodeIndexStart += dNodePerLevels[Lev].size();
		}
		{
			VDBNode Root = VDBNode::CreateInvalid();
			Root.Coord = CoordType(0, 0, 0);
			Root.ChildListOffset = 0;
			dNodePerLevels[VDBParams.RootLevel][0] = Root;
		}

		auto AssignChildPoolsKernel = [BrickYxX, VDBData = dVDBData] __device__(
										  int32_t Level, uint32_t NodeIndex) {
			auto&	  VDBParams = VDBData->VDBParams;
			CoordType Coord = VDBData->Node(Level, NodeIndex).Coord;

			int32_t	  ParentLevel = VDBParams.RootLevel;
			VDBNode	  Parent = VDBData->Node(ParentLevel, 0);
			CoordType ChildCoordInParent = VDBData->MapCoord(ParentLevel - 1, Level, Coord);
			uint32_t  ChildIndexInParent =
				VDBData->ChildIndexInParent(ParentLevel, ChildCoordInParent);
			while (ParentLevel != Level + 1)
			{
				Parent = VDBData->Node(
					ParentLevel - 1, VDBData->Child(ParentLevel, ChildIndexInParent, Parent));
				--ParentLevel;
				ChildCoordInParent = VDBData->MapCoord(ParentLevel - 1, Level, Coord)
					- Parent.Coord * VDBParams.ChildPerLevels[ParentLevel];
				ChildIndexInParent = VDBData->ChildIndexInParent(ParentLevel, ChildCoordInParent);
			}

			VDBData->Child(ParentLevel, ChildIndexInParent, Parent) = NodeIndex;
		};

		for (int32_t Lev = VDBParams.RootLevel - 1; Lev >= 0; --Lev)
		{
			thrust::for_each(thrust::cuda::par_nosync.on(getStream(EStream::VDB)),
				thrust::make_counting_iterator(uint32_t(0)),
				thrust::make_counting_iterator(static_cast<uint32_t>(dNodePerLevels[Lev].size())),
				[Lev, AssignChildPoolsKernel] __device__(
					uint32_t NodeIndex) { AssignChildPoolsKernel(Lev, NodeIndex); });
		}

#ifdef DEPTHBOX_DEBUG
		CUDA_CHECK(cudaStreamSynchronize(getStream(EStream::VDB)));

		for (int32_t Lev = VDBParams.RootLevel; Lev >= 0; --Lev)
		{
			std::cout << std::format("Lev: {}\n", Lev);
			CUDA_DEBUG_IN_CPU(dNodePerLevels[Lev]);
			if (Lev > 0)
			{
				CUDA_DEBUG_IN_CPU(dChildPerLevels[Lev - 1]);
			}
		}
#endif
	}

	RsdDataPF.Record(ResidentDataPerFrame::EEvent::BuildVDB, getStream(EStream::VDB));
}

void DepthBoxVDB::VolData::VDB::switchToFrame(uint32_t FrameIndex)
{
	uint32_t FrameNum = GetFrameNum();
	if (FrameIndex >= FrameNum)
	{
		std::cerr << std::format("FrameIndex:{} >= FrameNum:{}.", FrameIndex, FrameNum);
		dVDBDataCurrentFrame = nullptr;
		return;
	}

	uint32_t BrickNum = static_cast<uint32_t>(VDBParams.BrickPerVolume.z)
		* VDBParams.BrickPerVolume.y * VDBParams.BrickPerVolume.x;

	// Recycle and back to Atlas
	uint32_t ResidentFrameNum = ResidentIndices.size();
	{
		uint32_t ResidentFrameNumPriorToCurr = ResidentFrameNum;
		while (ResidentFrameNumPriorToCurr != 0 && [&]() {
			return ResidentDataPerFrames[ResidentIndices.front()].FrameIndex != FrameIndex;
		}())
		{
			uint32_t ResidentIndex = ResidentIndices.front();
			ResidentIndices.pop_front();
			auto& RsdDataPF = ResidentDataPerFrames[ResidentIndex];

			if (RsdDataPF.FrameIndex + FrameNum <= FrameIndex + MaxResidentFrameNum - 1)
			{
				// Should still be resident, move it to the tail of ResidentIndices
				ResidentIndices.emplace_back(ResidentIndex);
				RsdDataPF.ResidentIndicesItr = std::prev(ResidentIndices.end());
				--ResidentFrameNumPriorToCurr;
				continue;
			}

			// Mark it as NOT resident by swapping with the tail of ResidentDataPerFrames
			for (auto [BrickIndexWithFrame, AtlasBrickIndex] : RsdDataPF.BrickWithFrameToAtlasBrick)
			{
				BrickWithFrameToAtlasBrick[BrickIndexWithFrame] = kInvalidIndex;
				AtlasBrickToBrickWithFrame[AtlasBrickIndex] = kInvalidIndex;
				AvailableAtlasBrick.emplace_back(AtlasBrickIndex);
			}
			RsdDataPF.Invalidate();
			--ResidentFrameNum;
			std::swap(
				ResidentDataPerFrames[ResidentFrameNum], ResidentDataPerFrames[ResidentIndex]);
			*ResidentDataPerFrames[ResidentFrameNum].ResidentIndicesItr = ResidentIndex;

			--ResidentFrameNumPriorToCurr;
		}
	}

	// Allocate from Atlas and perform Transfer and Compuatation
	{
		uint32_t NewNeededFrameIndex = ResidentFrameNum == 0
			? FrameIndex
			: (ResidentDataPerFrames[ResidentIndices.back()].FrameIndex + 1) % FrameNum;
		uint32_t NewNeededNum = MaxResidentFrameNum - ResidentFrameNum;
		for (uint32_t NewNeededIndex = 0; NewNeededIndex < NewNeededNum; ++NewNeededIndex)
		{
			uint32_t ResidentIndex = ResidentFrameNum;
			++ResidentFrameNum;
			ResidentIndices.emplace_back(ResidentIndex);

			auto& RsdDataPF = ResidentDataPerFrames[ResidentIndex];
			RsdDataPF.FrameIndex = NewNeededFrameIndex;
			RsdDataPF.ResidentIndicesItr = std::prev(ResidentIndices.end());
			NewNeededFrameIndex = (NewNeededFrameIndex + 1) % FrameNum;

			auto& DataPF = DataPerFrames[RsdDataPF.FrameIndex];
			for (uint32_t BSKIndex = 0; BSKIndex < DataPF.BrickSortKeys.size(); ++BSKIndex)
			{
#ifdef DEPTHBOX_DEBUG
				if (AvailableAtlasBrick.empty())
				{
					throw std::exception("Algorithm Error!");
				}
#endif
				uint32_t BrickIndexWithFrame = [&]() {
					BrickSortKey	   BSKey = DataPF.BrickSortKeys[BSKIndex];
					CoordWithFrameType BrickCoordWithFrame;
					BrickCoordWithFrame.x = BSKey.LevelPosition.X;
					BrickCoordWithFrame.y = BSKey.LevelPosition.Y;
					BrickCoordWithFrame.z = BSKey.LevelPosition.Z;
					BrickCoordWithFrame.w = RsdDataPF.FrameIndex;

					return BrickCoordToIndex(BrickCoordWithFrame);
				}();

				uint32_t AtlasBrickIndex = AvailableAtlasBrick.back();
				AvailableAtlasBrick.pop_back();

				RsdDataPF.BrickWithFrameToAtlasBrick.emplace(BrickIndexWithFrame, AtlasBrickIndex);
				BrickWithFrameToAtlasBrick[BrickIndexWithFrame] = AtlasBrickIndex;
				AtlasBrickToBrickWithFrame[AtlasBrickIndex] = BrickIndexWithFrame;
			}

			transferBrickDataToAtlas(ResidentIndex);
			updateDepthBox(ResidentIndex);
			transferBrickDataToCPU(ResidentIndex);
			buildVDB(ResidentIndex);
		}
	}

	// Popup the Device VDB Data if it is ready
	if (!ResidentIndices.empty()
		&& !ResidentDataPerFrames[ResidentIndices.front()].BrickWithFrameToAtlasBrick.empty())
	{
		auto& RsdDataPF = ResidentDataPerFrames[ResidentIndices.front()];
		RsdDataPF.Wait(getStream(EStream::Host), ResidentDataPerFrame::EEvent::BuildVDB);

		PopupFrameParams.OutVDB = this;
		PopupFrameParams.InRsdDataPF = &RsdDataPF;
		CUDA_CHECK(cudaLaunchHostFunc(getStream(EStream::Host), Popup, &PopupFrameParams));
	}
	else
	{
		dVDBDataCurrentFrame = nullptr;
	}
}

void DepthBoxVDB::VolData::VDB::invalidate()
{
	waitForAllStream();

	DataPerFrames.clear();

	MaxResidentFrameNum = 0;
	ResidentIndices.clear();
	ResidentDataPerFrames.clear();

	dVDBDataCurrentFrame = nullptr;

	AvailableAtlasBrick.clear();
	AtlasBrickToBrickWithFrame.clear();
	BrickWithFrameToAtlasBrick.clear();
}

void DepthBoxVDB::VolData::VDB::waitForAllStream()
{
	for (uint32_t i = 0; i < static_cast<uint32_t>(EStream::Max); ++i)
	{
		CUDA_CHECK(cudaStreamSynchronize(Streams[i]));
	}
}
