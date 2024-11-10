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

#define ENABLE_CUDA_DEBUG_IN_CPU

std::shared_ptr<DepthBoxVDB::VolData::IVDBDataProvider>
DepthBoxVDB::VolData::IVDBDataProvider::Create(const CreateParameters& Params)
{
	return std::make_shared<VDBProvider>(Params);
}

std::shared_ptr<DepthBoxVDB::VolData::IVDBBuilder> DepthBoxVDB::VolData::IVDBBuilder::Create(
	const CreateParameters& Params)
{
	return std::make_shared<VDBBuilder>(Params);
}

union BrickSortKey
{
	uint64_t Key;
	struct
	{
		uint64_t Z : 20;
		uint64_t Y : 20;
		uint64_t X : 20;
		uint64_t Level : 4;
	} LevelPosition;

	__host__ constexpr static BrickSortKey Invalid()
	{
		BrickSortKey Ret{ std::numeric_limits<uint64_t>::max() };
		return Ret;
	}

	__host__ __device__ bool operator==(const BrickSortKey& Other) const
	{
		return Key == Other.Key;
	}
	__host__ __device__ bool operator<(const BrickSortKey& Other) const { return Key < Other.Key; }
};

DepthBoxVDB::VolData::VDBProvider::VDBProvider(const CreateParameters& Params)
	: VDBParams(Params.VDBParams), MaxAllowedGPUMemoryInGB(Params.MaxAllowedGPUMemoryInGB)
{
	int DeviceNum = 0;
	CUDA_CHECK(cudaGetDeviceCount(&DeviceNum));
	assert(DeviceNum > 0);

	cudaDeviceProp Prop;
	CUDA_CHECK(cudaGetDeviceProperties(&Prop, 0));

	CUDA_CHECK(cudaStreamCreate(&Stream));

	relayoutRAWVolume(Params);
}

DepthBoxVDB::VolData::VDBProvider::~VDBProvider() {}

void DepthBoxVDB::VolData::VDBProvider::relayoutRAWVolume(const CreateParameters& Params)
{
	uint32_t VoxelYxX = VDBParams.VoxelPerVolume.x * VDBParams.VoxelPerVolume.y;
	uint32_t VoxelYxXPerAtlasBrick = VDBParams.VoxelPerAtlasBrick * VDBParams.VoxelPerAtlasBrick;
	uint32_t BrickYxX = VDBParams.BrickPerVolume.x * VDBParams.BrickPerVolume.y;
	uint32_t BrickNum = BrickYxX * VDBParams.BrickPerVolume.z;
	uint32_t VoxelNumPerBrickWithApron =
		static_cast<uint32_t>(VDBParams.ChildPerLevels[0] + 2 * VDBParams.ApronWidth)
		* (VDBParams.ChildPerLevels[0] + 2 * VDBParams.ApronWidth)
		* (VDBParams.ChildPerLevels[0] + 2 * VDBParams.ApronWidth);
	uint32_t VoxelNumPerAtlasBrick = VoxelYxXPerAtlasBrick * VDBParams.VoxelPerAtlasBrick;
	uint32_t VoxelApronOffset = VDBParams.ApronAndDepthWidth - VDBParams.ApronWidth;

	std::vector<std::future<void>> Futures(BrickNum);
	BrickToAtlasBrick.assign(BrickNum, CoordType(InvalidCoordValue));
	dBrickToAtlasBrick.resize(BrickNum);
	BrickedData.resize(SizeOfVoxelType(VDBParams.VoxelType) * BrickNum * VoxelNumPerAtlasBrick);

	auto Assign = [&]<typename T>(T* Dst, T* Src, const CoordType& BrickCoord) {
		auto Sample = [&](CoordType Coord) -> T {
			Coord = glm::clamp(Coord, CoordType(0), VDBParams.VoxelPerVolume - 1);
			return Src[Coord.z * VoxelYxX + Coord.y * VDBParams.VoxelPerVolume.x + Coord.x];
		};

		CoordType CoordMin = BrickCoord * VDBParams.ChildPerLevels[0];
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
					T Scalar = Sample(CoordMin + dCoord - VDBParams.ApronAndDepthWidth);
					DstPitchPtr[dCoord.x] = Scalar;

					for (uint32_t RngIdx = 0; RngIdx < Params.EmptyScalarRangeNum; ++RngIdx)
					{
						glm::vec2 Range = Params.EmptyScalarRanges[RngIdx];
						if (Range[0] <= Scalar && Scalar <= Range[1])
						{
							++EmptyVoxelNum;
						}
					}
				}

				DstPitchPtr += VDBParams.VoxelPerAtlasBrick;
			}
		}

		if (EmptyVoxelNum != VoxelNumPerBrickWithApron)
		{
			BrickToAtlasBrick[BrickIndex] = BrickCoord;
		}
	};

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
								reinterpret_cast<uint8_t*>(Params.RAWVolumeData), BrickCoord);
						break;
					case EVoxelType::Float32:
						Futures[BrickIndex] =
							std::async(Assign, reinterpret_cast<float*>(BrickedData.data()),
								reinterpret_cast<float*>(Params.RAWVolumeData), BrickCoord);
						break;
				}

				++BrickIndex;
			}

	for (auto& Future : Futures)
	{
		Future.wait();
	}
	uint32_t AtlasBrickIndex = 0;
	for (uint32_t BrickIndex = 0; BrickIndex < BrickNum; ++BrickIndex)
	{
		if (BrickToAtlasBrick[BrickIndex] == CoordType(InvalidCoordValue))
			continue;

		CoordType AtlasBrickCoord;
		AtlasBrickCoord.z = AtlasBrickIndex / BrickYxX;
		uint32_t Tmp = AtlasBrickCoord.z * BrickYxX;
		AtlasBrickCoord.y = (AtlasBrickIndex - Tmp) / VDBParams.BrickPerVolume.x;
		AtlasBrickCoord.x = AtlasBrickIndex - Tmp - AtlasBrickCoord.y * VDBParams.BrickPerVolume.x;

		BrickToAtlasBrick[BrickIndex] = AtlasBrickCoord;

		++AtlasBrickIndex;
	}
	ValidBrickNum = AtlasBrickIndex;

	resizeAtlas();

	uint32_t BrickNumPerAtlas =
		static_cast<uint32_t>(BrickPerAtlas.z) * BrickPerAtlas.y * BrickPerAtlas.x;
	AtlasBrickToBrick.assign(BrickNumPerAtlas, CoordType(InvalidCoordValue));
	dAtlasBrickToBrick.resize(BrickNumPerAtlas);

	AtlasBrickIndex = 0;
	for (uint32_t BrickIndex = 0; BrickIndex < BrickNum; ++BrickIndex)
	{
		if (BrickToAtlasBrick[BrickIndex] == CoordType(InvalidCoordValue))
			continue;

		CoordType BrickCoord;
		BrickCoord.z = BrickIndex / BrickYxX;
		uint32_t Tmp = BrickCoord.z * BrickYxX;
		BrickCoord.y = (BrickIndex - Tmp) / VDBParams.BrickPerVolume.x;
		BrickCoord.x = BrickIndex - Tmp - BrickCoord.y * VDBParams.BrickPerVolume.x;

		AtlasBrickToBrick[AtlasBrickIndex] = BrickCoord;

		++AtlasBrickIndex;
	}

	updateAtlas();
	CUDA_CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(dAtlasBrickToBrick.data()),
		AtlasBrickToBrick.data(), sizeof(CoordType) * AtlasBrickToBrick.size(),
		cudaMemcpyHostToDevice, Stream));
	CUDA_CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(dBrickToAtlasBrick.data()),
		BrickToAtlasBrick.data(), sizeof(CoordType) * BrickToAtlasBrick.size(),
		cudaMemcpyHostToDevice, Stream));
}

void DepthBoxVDB::VolData::VDBProvider::resizeAtlas()
{
	CoordType NeededVoxelPerAtlas;
	// Compute NeededVoxelPerAtlas
	{
		BrickPerAtlas = VDBParams.BrickPerVolume;
		BrickPerAtlas.z = 0;

		while (static_cast<uint32_t>(BrickPerAtlas.z) * BrickPerAtlas.y * BrickPerAtlas.x
			< ValidBrickNum)
		{
			++BrickPerAtlas.z;
		}
		NeededVoxelPerAtlas = BrickPerAtlas * VDBParams.VoxelPerAtlasBrick;

		size_t MaxAllowedGPUMemoryInByte =
			SizeOfVoxelType(VDBParams.VoxelType) * MaxAllowedGPUMemoryInGB * (1 << 30);
		if (SizeOfVoxelType(VDBParams.VoxelType) * NeededVoxelPerAtlas.x * NeededVoxelPerAtlas.y
				* NeededVoxelPerAtlas.z
			> MaxAllowedGPUMemoryInByte)
		{
			std::cerr << "MaxAllowedGPUMemoryInGB is too small.\n";
			return;
		}

		if (AtlasArray && [&]() {
				auto Extent = AtlasArray->GetExtent();
				return Extent.width == NeededVoxelPerAtlas.x
					&& Extent.height == NeededVoxelPerAtlas.y
					&& Extent.depth == NeededVoxelPerAtlas.z;
			}())
			return;
	}

	// Resize Atlas
	{
		cudaChannelFormatDesc ChannelDesc;
		switch (VDBParams.VoxelType)
		{
			case EVoxelType::UInt8:
				ChannelDesc = cudaCreateChannelDesc<uint8_t>();
				break;
			case EVoxelType::Float32:
				ChannelDesc = cudaCreateChannelDesc<float>();
				break;
			default:
				return;
		}
		AtlasArray = std::make_shared<CUDA::Array>(ChannelDesc, NeededVoxelPerAtlas);
	}

	// Create Texture and Surface
	AtlasTexture = std::make_unique<CUDA::Texture>(AtlasArray);
	AtlasSurface = std::make_unique<CUDA::Surface>(AtlasArray);
}

void DepthBoxVDB::VolData::VDBProvider::updateAtlas()
{
	uint32_t BrickYxX = VDBParams.BrickPerVolume.x * VDBParams.BrickPerVolume.y;
	uint32_t VoxelNumPerAtlasBrick = static_cast<uint32_t>(VDBParams.VoxelPerAtlasBrick)
		* VDBParams.VoxelPerAtlasBrick * VDBParams.VoxelPerAtlasBrick;

	auto Transfer = [&]<typename T>(T* Src, uint32_t AtlasBrickIndex) {
		CoordType BrickCoord = AtlasBrickToBrick[AtlasBrickIndex];
		uint32_t  BrickIndex = static_cast<uint32_t>(BrickCoord.z) * BrickYxX
			+ BrickCoord.y * VDBParams.BrickPerVolume.x + BrickCoord.x;

		CoordType AtlasBrickCoord;
		AtlasBrickCoord.z = AtlasBrickIndex / BrickYxX;
		uint32_t Tmp = AtlasBrickCoord.z * BrickYxX;
		AtlasBrickCoord.y = (AtlasBrickIndex - Tmp) / VDBParams.BrickPerVolume.x;
		AtlasBrickCoord.x = AtlasBrickIndex - Tmp - AtlasBrickCoord.y * VDBParams.BrickPerVolume.x;

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

		CUDA_CHECK(cudaMemcpy3DAsync(&MemCpyParams, Stream));
	};

	for (uint32_t AtlasBrickIndex = 0; AtlasBrickIndex < AtlasBrickToBrick.size();
		 ++AtlasBrickIndex)
	{
		if (AtlasBrickToBrick[AtlasBrickIndex] == CoordType(InvalidCoordValue))
			continue;

		switch (VDBParams.VoxelType)
		{
			case EVoxelType::UInt8:
				Transfer(reinterpret_cast<uint8_t*>(BrickedData.data()), AtlasBrickIndex);
				break;
			case EVoxelType::Float32:
				Transfer(reinterpret_cast<float*>(BrickedData.data()), AtlasBrickIndex);
				break;
		}
	}
}

DepthBoxVDB::VolData::VDBBuilder::VDBBuilder(const CreateParameters& Params)
{
	int DeviceNum = 0;
	CUDA_CHECK(cudaGetDeviceCount(&DeviceNum));
	assert(DeviceNum > 0);

	cudaDeviceProp Prop;
	CUDA_CHECK(cudaGetDeviceProperties(&Prop, 0));

	CUDA_CHECK(cudaStreamCreate(&Stream));
}

DepthBoxVDB::VolData::VDBBuilder::~VDBBuilder()
{
	if (dData)
	{
		CUDA_CHECK(cudaFree(dData));
	}
}

#ifdef ENABLE_CUDA_DEBUG_IN_CPU
template <typename T> void DebugInCPU(const thrust::device_vector<T>& dVector, const char* Name)
{
	thrust::host_vector<T> hVetcor = dVector;
	std ::string		   DebugMsg = std::format("CUDA Debug {}:\n\t", Name);
	uint32_t			   Index = 0;
	uint32_t			   Column = 0;
	for (auto Itr = hVetcor.begin(); Itr != hVetcor.end(); ++Itr)
	{
		if (Column == 10)
		{
			DebugMsg.append("\n\t");
			Column = 0;
		}

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

		++Column;
		++Index;
	}
	DebugMsg.push_back('\n');

	std::cout << DebugMsg;
}

	#define CUDA_DEBUG_IN_CPU(X) DebugInCPU(X, #X)

#endif

void DepthBoxVDB::VolData::VDBBuilder::FullBuild(const FullBuildParameters& Params)
{
	CUDA_CHECK(cudaStreamSynchronize(Stream));

	Provider = std::static_pointer_cast<VDBProvider>(Params.Provider);
	CUDA_CHECK(cudaStreamSynchronize(Provider->Stream));

	if (!Provider->AtlasSurface)
	{
		std::cerr << "Invalid Provider.\n";
		return;
	}

	auto&	 VDBParams = Provider->VDBParams;
	uint32_t BrickYxX = VDBParams.BrickPerVolume.x * VDBParams.BrickPerVolume.y;
	uint32_t BrickNum = BrickYxX * VDBParams.BrickPerVolume.z;
	uint32_t ValidBrickNum = Provider->ValidBrickNum;
	uint32_t VoxelNumPerBrick =
		VDBParams.ChildPerLevels[0] * VDBParams.ChildPerLevels[0] * VDBParams.ChildPerLevels[0];

	// 1. Assign Brick Sort Keys to non-empty Brick
	// 2. Sort Brick Sort Keys
	thrust::device_vector<BrickSortKey> dBrickSortKeys(
		(VDBParams.RootLevel + 1) * BrickNum, BrickSortKey::Invalid());
	{
		thrust::device_vector<glm::vec2> dEmptyScalarRanges(Params.EmptyScalarRangeNum);
		CUDA_CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(dEmptyScalarRanges.data()),
			Params.EmptyScalarRanges, sizeof(glm::vec2) * Params.EmptyScalarRangeNum,
			cudaMemcpyHostToDevice, Stream));

		auto AssignBrickKeysKernel =
			[ValidBrickNum, BrickSortKeys = thrust::raw_pointer_cast(dBrickSortKeys.data()),
				AtlasBrickToBrick = thrust::raw_pointer_cast(Provider->dAtlasBrickToBrick.data()),
				AtlasSurface = Provider->AtlasSurface->Get(), VoxelNumPerBrick, BrickYxX, BrickNum,
				VDBParams] __device__(const glm::uvec3& AtlasBrickIndex) {
				if (AtlasBrickIndex.x >= ValidBrickNum)
					return;

				CoordType BrickCoord = AtlasBrickToBrick[AtlasBrickIndex.x];
				CoordType CoordMin = static_cast<CoordType>(BrickCoord)
						* static_cast<CoordValueType>(VDBParams.VoxelPerAtlasBrick)
					+ static_cast<CoordValueType>(VDBParams.ApronAndDepthWidth);

				BrickSortKey Key;
				Key.LevelPosition.Level = 0;
				Key.LevelPosition.X = BrickCoord.x;
				Key.LevelPosition.Y = BrickCoord.y;
				Key.LevelPosition.Z = BrickCoord.z;
				uint32_t FlatIdx = BrickCoord.z * BrickYxX
					+ BrickCoord.y * VDBParams.BrickPerVolume.x + BrickCoord.x;
				BrickSortKeys[FlatIdx] = Key;

				for (int32_t Lev = 1; Lev <= VDBParams.RootLevel; ++Lev)
				{
					Key.LevelPosition.Level = Lev;
					Key.LevelPosition.X /= VDBParams.ChildPerLevels[Lev];
					Key.LevelPosition.Y /= VDBParams.ChildPerLevels[Lev];
					Key.LevelPosition.Z /= VDBParams.ChildPerLevels[Lev];

					FlatIdx = Lev * BrickNum + BrickCoord.z * BrickYxX
						+ BrickCoord.y * VDBParams.BrickPerVolume.x + BrickCoord.x;
					BrickSortKeys[FlatIdx] = Key;
				}
			};

		const dim3 ThreadPerBlock1D = { CUDA::ThreadPerBlockX1D, 1, 1 };

		dim3 BlockPerGrid;
		BlockPerGrid.x = (ValidBrickNum + ThreadPerBlock1D.x - 1) / ThreadPerBlock1D.x;
		BlockPerGrid.y = 1;
		BlockPerGrid.z = 1;
		CUDA::ParallelFor(BlockPerGrid, ThreadPerBlock1D, AssignBrickKeysKernel, Stream);

		thrust::sort(thrust::cuda::par.on(Stream), dBrickSortKeys.begin(), dBrickSortKeys.end());

#ifdef ENABLE_CUDA_DEBUG_IN_CPU
		CUDA_CHECK(cudaStreamSynchronize(Stream));
		CUDA_DEBUG_IN_CPU(dEmptyScalarRanges);
		CUDA_DEBUG_IN_CPU(dBrickSortKeys);
#endif
	}

	// Compact Brick Sort Keys
	{
		uint32_t ValidNum = [&]() {
			auto ItrInvalid = thrust::lower_bound(thrust::cuda::par.on(Stream),
				dBrickSortKeys.begin(), dBrickSortKeys.end(), BrickSortKey::Invalid());
			return thrust::distance(dBrickSortKeys.begin(), ItrInvalid);
		}();
		auto dDiffs = CUDA::Difference<uint32_t>(dBrickSortKeys, ValidNum, Stream);
		dBrickSortKeys = CUDA::Compact(dBrickSortKeys, dDiffs, ValidNum, Stream);

#ifdef ENABLE_CUDA_DEBUG_IN_CPU
		CUDA_CHECK(cudaStreamSynchronize(Stream));
		CUDA_DEBUG_IN_CPU(dBrickSortKeys);
#endif
	}

	// Allocate Node and Child Pools
	{
		uint32_t StartCurrLev = 0;
		for (int32_t Lev = 0; Lev <= VDBParams.RootLevel; ++Lev)
		{
			BrickSortKey KeyNextLev;
			KeyNextLev.LevelPosition.Level = Lev + 1;
			KeyNextLev.LevelPosition.X = KeyNextLev.LevelPosition.Y = KeyNextLev.LevelPosition.Z =
				0;
			uint32_t NumCurrLev = Lev == VDBParams.RootLevel ? 1 : [&]() {
				auto ItrInvalid = thrust::lower_bound(thrust::cuda::par.on(Stream),
					dBrickSortKeys.begin() + StartCurrLev, dBrickSortKeys.end(), KeyNextLev);
				return thrust::distance(dBrickSortKeys.begin(), ItrInvalid);
			}();
			StartCurrLev += NumCurrLev;

			dNodePerLevels[Lev].resize(NumCurrLev, VDBNode::CreateInvalid());

			if (Lev > 0)
			{
				uint64_t ChildCurrLev = VDBParams.ChildPerLevels[Lev];
				dChildPerLevels[Lev - 1].resize(
					dNodePerLevels[Lev].size() * ChildCurrLev * ChildCurrLev * ChildCurrLev,
					VDBData::InvalidChild);
			}
		}
	}

	// Upload
	VDBData Data;
	Data.VDBParams = VDBParams;
	Data.AtlasTexture = Provider->AtlasTexture->Get();
	Data.AtlasSurface = Provider->AtlasSurface->Get();
	for (int32_t Lev = 0; Lev <= VDBParams.RootLevel; ++Lev)
	{
		Data.NodePerLevels[Lev] = thrust::raw_pointer_cast(dNodePerLevels[Lev].data());
		if (Lev > 0)
		{
			Data.ChildPerLevels[Lev - 1] =
				thrust::raw_pointer_cast(dChildPerLevels[Lev - 1].data());
		}
	}
	if (!dData)
	{
		CUDA_CHECK(cudaMalloc(&dData, sizeof(VDBData)));
	}
	CUDA_CHECK(cudaMemcpy(dData, &Data, sizeof(VDBData), cudaMemcpyHostToDevice));

	// Assign Node and Child Pools
	{
		auto AssignNodePoolsKernel =
			[BrickYxX, Data = dData,
				BrickToAtlasBrick = thrust::raw_pointer_cast(Provider->dBrickToAtlasBrick.data()),
				BrickSortKeys =
					thrust::raw_pointer_cast(dBrickSortKeys.data())] __device__(int32_t Level,
				uint64_t NodeIndexStart, uint32_t NodeIndex) {
				auto& VDBParams = Data->VDBParams;

				VDBNode		 Node;
				BrickSortKey SortKey = BrickSortKeys[NodeIndexStart + NodeIndex];
				Node.Coord = CoordType(
					SortKey.LevelPosition.X, SortKey.LevelPosition.Y, SortKey.LevelPosition.Z);

				if (Level == 0)
				{
					uint32_t BrickIndex = Node.Coord.z * BrickYxX
						+ Node.Coord.y * VDBParams.BrickPerVolume.x + Node.Coord.x;
					Node.CoordInAtlas = BrickToAtlasBrick[BrickIndex];
				}
				else
				{
					int32_t ChildCurrLev = VDBParams.ChildPerLevels[Level];
					Node.ChildListOffset = static_cast<uint64_t>(NodeIndex) * ChildCurrLev
						* ChildCurrLev * ChildCurrLev;
				}

				Data->Node(Level, NodeIndex) = Node;
			};

		uint64_t NodeIndexStart = 0;
		for (int32_t Lev = 0; Lev < VDBParams.RootLevel; ++Lev)
		{
			thrust::for_each(thrust::cuda::par.on(Stream),
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

		auto AssignChildPoolsKernel = [BrickYxX, Data = dData] __device__(
										  int32_t Level, uint32_t NodeIndex) {
			auto&	  VDBParams = Data->VDBParams;
			CoordType Coord = Data->Node(Level, NodeIndex).Coord;

			int32_t ParentLevel = Level + 1;
			VDBNode Parent = Data->Node(ParentLevel, Data->CoordInParentLevel(ParentLevel, Coord));
			CoordType ChildCoordInParent = Data->ChildCoordInParent(ParentLevel, Coord);
			uint32_t ChildIndexInParent = Data->ChildIndexInParent(ParentLevel, ChildCoordInParent);

			Data->Child(ParentLevel, ChildIndexInParent, Parent) = NodeIndex;
		};

		for (int32_t Lev = VDBParams.RootLevel - 1; Lev >= 0; --Lev)
		{
			thrust::for_each(thrust::cuda::par.on(Stream),
				thrust::make_counting_iterator(uint32_t(0)),
				thrust::make_counting_iterator(static_cast<uint32_t>(dNodePerLevels[Lev].size())),
				[Lev, AssignChildPoolsKernel] __device__(
					uint32_t NodeIndex) { AssignChildPoolsKernel(Lev, NodeIndex); });
		}

#ifdef ENABLE_CUDA_DEBUG_IN_CPU
		CUDA_CHECK(cudaStreamSynchronize(Stream));
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
}

DepthBoxVDB::VolData::VDBData* DepthBoxVDB::VolData::VDBBuilder::GetDeviceData() const
{
	if (!Provider)
		return nullptr;

	CUDA_CHECK(cudaStreamSynchronize(Provider->Stream));
	CUDA_CHECK(cudaStreamSynchronize(Stream));
	return dData;
}
