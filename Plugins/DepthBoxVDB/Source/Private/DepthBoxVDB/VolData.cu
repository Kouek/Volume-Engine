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

DepthBoxVDB::VolData::VDBBuilder::VDBBuilder(const CreateParameters& Params)
{
	int DeviceNum = 0;
	CUDA_CHECK(cudaGetDeviceCount(&DeviceNum));
	assert(DeviceNum > 0);

	cudaDeviceProp Prop;
	CUDA_CHECK(cudaGetDeviceProperties(&Prop, 0));

	CUDA_CHECK(cudaStreamCreate(&AtlasStream));
	CUDA_CHECK(cudaStreamCreate(&NodeStream));
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

void DepthBoxVDB::VolData::VDBBuilder::FullBuild(const FullBuildParameters& Params)
{
	CUDA_CHECK(cudaStreamSynchronize(NodeStream));

	VDBParams = Params.VDBParams;
	MaxAllowedGPUMemoryInGB = Params.MaxAllowedGPUMemoryInGB;

	// Update Atlas
	relayoutRAWVolume(Params);
	if (ValidBrickNum == 0)
	{
		std::cout << "No valid bricks.\n";
		return;
	}

	UpdateDepthBoxAsync({ .EmptyScalarRanges = Params.EmptyScalarRanges,
		.EmptyScalarRangeNum = Params.EmptyScalarRangeNum });

	uint32_t BrickYxX = VDBParams.BrickPerVolume.x * VDBParams.BrickPerVolume.y;

	// 1. Assign Brick Sort Keys to non-empty Brick
	// 2. Sort Brick Sort Keys
	thrust::device_vector<BrickSortKey> dBrickSortKeys(
		(VDBParams.RootLevel + 1) * ValidBrickNum, BrickSortKey::Invalid());
	{
		auto AssignBrickKeysKernel =
			[BrickSortKeys = thrust::raw_pointer_cast(dBrickSortKeys.data()),
				AtlasBrickToBrick = thrust::raw_pointer_cast(dAtlasBrickToBrick.data()), BrickYxX,
				ValidBrickNum = ValidBrickNum,
				VDBParams = VDBParams] __device__(uint32_t AtlasBrickIndex) {
				CoordType BrickCoord = AtlasBrickToBrick[AtlasBrickIndex];

				BrickSortKey Key;
				Key.LevelPosition.Level = 0;
				Key.LevelPosition.X = BrickCoord.x;
				Key.LevelPosition.Y = BrickCoord.y;
				Key.LevelPosition.Z = BrickCoord.z;
				uint32_t FlatIdx = AtlasBrickIndex;
				BrickSortKeys[FlatIdx] = Key;

				for (int32_t Lev = 1; Lev <= VDBParams.RootLevel; ++Lev)
				{
					Key.LevelPosition.Level = Lev;
					Key.LevelPosition.X /= VDBParams.ChildPerLevels[Lev];
					Key.LevelPosition.Y /= VDBParams.ChildPerLevels[Lev];
					Key.LevelPosition.Z /= VDBParams.ChildPerLevels[Lev];

					FlatIdx = Lev * ValidBrickNum + AtlasBrickIndex;
					BrickSortKeys[FlatIdx] = Key;
				}
			};

		thrust::for_each(thrust::cuda::par.on(NodeStream),
			thrust::make_counting_iterator(uint32_t(0)),
			thrust::make_counting_iterator(ValidBrickNum), AssignBrickKeysKernel);

		thrust::sort(
			thrust::cuda::par.on(NodeStream), dBrickSortKeys.begin(), dBrickSortKeys.end());

#ifdef ENABLE_CUDA_DEBUG_IN_CPU
		CUDA_CHECK(cudaStreamSynchronize(NodeStream));
		CUDA_DEBUG_IN_CPU(dBrickSortKeys);
#endif
	}

	// Compact Brick Sort Keys
	{
		auto dDiffs = CUDA::Difference<uint32_t>(dBrickSortKeys, 0, NodeStream);
		dBrickSortKeys = CUDA::Compact(dBrickSortKeys, dDiffs, uint32_t(0), NodeStream);

#ifdef ENABLE_CUDA_DEBUG_IN_CPU
		CUDA_CHECK(cudaStreamSynchronize(NodeStream));
		CUDA_DEBUG_IN_CPU(dBrickSortKeys);
#endif
	}

	// Allocate Node and Child Pools
	{
		uint32_t StartCurrLev = 0;
		for (int32_t Lev = 0; Lev <= VDBParams.RootLevel; ++Lev)
		{
			uint32_t NumCurrLev = Lev == VDBParams.RootLevel ? 1 : [&]() {
				BrickSortKey KeyNextLev;
				KeyNextLev.LevelPosition.Level = Lev + 1;
				KeyNextLev.LevelPosition.X = KeyNextLev.LevelPosition.Y =
					KeyNextLev.LevelPosition.Z = 0;

				auto ItrCurrLev = dBrickSortKeys.begin() + StartCurrLev;
				auto ItrNextLev = thrust::lower_bound(
					thrust::cuda::par.on(NodeStream), ItrCurrLev, dBrickSortKeys.end(), KeyNextLev);

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
	VDBData Data;
	Data.VDBParams = VDBParams;
	Data.AtlasTexture = AtlasTexture->Get();
	Data.AtlasSurface = AtlasSurface->Get();
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
		auto AssignNodePoolsKernel = [BrickYxX, Data = dData,
										 BrickToAtlasBrick =
											 thrust::raw_pointer_cast(dBrickToAtlasBrick.data()),
										 BrickSortKeys = thrust::raw_pointer_cast(
											 dBrickSortKeys.data())] __device__(int32_t Level,
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
				Node.ChildListOffset =
					static_cast<uint64_t>(NodeIndex) * ChildCurrLev * ChildCurrLev * ChildCurrLev;
			}

			Data->Node(Level, NodeIndex) = Node;
		};

		uint64_t NodeIndexStart = 0;
		for (int32_t Lev = 0; Lev < VDBParams.RootLevel; ++Lev)
		{
			thrust::for_each(thrust::cuda::par.on(NodeStream),
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

			int32_t	  ParentLevel = VDBParams.RootLevel;
			VDBNode	  Parent = Data->Node(ParentLevel, 0);
			CoordType ChildCoordInParent = Data->MapCoord(ParentLevel - 1, Level, Coord);
			uint32_t ChildIndexInParent = Data->ChildIndexInParent(ParentLevel, ChildCoordInParent);
			while (ParentLevel != Level + 1)
			{
				Parent = Data->Node(
					ParentLevel - 1, Data->Child(ParentLevel, ChildIndexInParent, Parent));
				--ParentLevel;
				ChildCoordInParent = Data->MapCoord(ParentLevel - 1, Level, Coord)
					- Parent.Coord * VDBParams.ChildPerLevels[ParentLevel];
				ChildIndexInParent = Data->ChildIndexInParent(ParentLevel, ChildCoordInParent);
			}

			Data->Child(ParentLevel, ChildIndexInParent, Parent) = NodeIndex;
		};

		for (int32_t Lev = VDBParams.RootLevel - 1; Lev >= 0; --Lev)
		{
			thrust::for_each(thrust::cuda::par.on(NodeStream),
				thrust::make_counting_iterator(uint32_t(0)),
				thrust::make_counting_iterator(static_cast<uint32_t>(dNodePerLevels[Lev].size())),
				[Lev, AssignChildPoolsKernel] __device__(
					uint32_t NodeIndex) { AssignChildPoolsKernel(Lev, NodeIndex); });
		}

#ifdef ENABLE_CUDA_DEBUG_IN_CPU
		CUDA_CHECK(cudaStreamSynchronize(NodeStream));
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

void DepthBoxVDB::VolData::VDBBuilder::UpdateDepthBoxAsync(const UpdateDepthBoxParameters& Params)
{
	if (!AtlasSurface)
	{
		std::cerr << "Invalid Atlas.\n";
		return;
	}

	switch (VDBParams.VoxelType)
	{
		case EVoxelType::UInt8:
			updateDepthBoxAsync<uint8_t>(Params);
			break;
		case EVoxelType::UInt16:
			updateDepthBoxAsync<uint16_t>(Params);
			break;
		case EVoxelType::Float32:
			updateDepthBoxAsync<float>(Params);
			break;
		default:
			assert(false);
	}
}

template <typename VoxelType>
void DepthBoxVDB::VolData::VDBBuilder::updateDepthBoxAsync(const UpdateDepthBoxParameters& Params)
{
	uint32_t BrickYxXPerAtlas = BrickPerAtlas.x * BrickPerAtlas.y;
	uint32_t VoxelYxXPerBrick = VDBParams.ChildPerLevels[0] * VDBParams.ChildPerLevels[0];
	uint32_t BrickNumPerAtlas =
		static_cast<uint32_t>(BrickPerAtlas.z) * BrickPerAtlas.y * BrickPerAtlas.x;
	uint32_t DepthVoxelNumPerAtlas = BrickNumPerAtlas * 6 * VoxelYxXPerBrick;

	thrust::device_vector<glm::vec2> dEmptyScalarRanges(Params.EmptyScalarRangeNum);
	// Sync here since EmptyScalarRanges cannot be maintained by this
	CUDA_CHECK(
		cudaMemcpy(thrust::raw_pointer_cast(dEmptyScalarRanges.data()), Params.EmptyScalarRanges,
			sizeof(glm::vec2) * Params.EmptyScalarRangeNum, cudaMemcpyHostToDevice));

	auto UpdateKernel = [DepthVoxelNumPerAtlas, VoxelYxXPerBrick, BrickYxXPerAtlas,
							BrickPerAtlas = BrickPerAtlas,
							EmptyScalarRangeNum = Params.EmptyScalarRangeNum,
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

	thrust::for_each(thrust::cuda::par.on(AtlasStream), thrust::make_counting_iterator(uint32_t(0)),
		thrust::make_counting_iterator(DepthVoxelNumPerAtlas), UpdateKernel);
}
template void DepthBoxVDB::VolData::VDBBuilder::updateDepthBoxAsync<uint8_t>(
	const UpdateDepthBoxParameters& Params);
template void DepthBoxVDB::VolData::VDBBuilder::updateDepthBoxAsync<uint16_t>(
	const UpdateDepthBoxParameters& Params);
template void DepthBoxVDB::VolData::VDBBuilder::updateDepthBoxAsync<float>(
	const UpdateDepthBoxParameters& Params);

DepthBoxVDB::VolData::VDBData* DepthBoxVDB::VolData::VDBBuilder::GetDeviceData() const
{
	if (ValidBrickNum == 0)
		return nullptr;

	CUDA_CHECK(cudaStreamSynchronize(AtlasStream));
	CUDA_CHECK(cudaStreamSynchronize(NodeStream));
	return dData;
}

void DepthBoxVDB::VolData::VDBBuilder::relayoutRAWVolume(const FullBuildParameters& Params)
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
	BrickToAtlasBrick.assign(BrickNum, CoordType(kInvalidCoordValue));
	dBrickToAtlasBrick.resize(BrickNum);
	BrickedData.resize(SizeOfVoxelType(VDBParams.VoxelType) * BrickNum * VoxelNumPerAtlasBrick);

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

					for (uint32_t RngIdx = 0; RngIdx < Params.EmptyScalarRangeNum; ++RngIdx)
					{
						glm::vec2 Range = Params.EmptyScalarRanges[RngIdx];
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
			BrickToAtlasBrick[BrickIndex] = BrickCoord;
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
							Futures[BrickIndex] = std::async(Assign,
								reinterpret_cast<uint8_t*>(BrickedData.data()),
								reinterpret_cast<const uint8_t*>(Params.RAWVolumeData), BrickCoord);
							break;
						case EVoxelType::UInt16:
							Futures[BrickIndex] =
								std::async(Assign, reinterpret_cast<uint16_t*>(BrickedData.data()),
									reinterpret_cast<const uint16_t*>(Params.RAWVolumeData),
									BrickCoord);
							break;
						case EVoxelType::Float32:
							Futures[BrickIndex] = std::async(Assign,
								reinterpret_cast<float*>(BrickedData.data()),
								reinterpret_cast<const float*>(Params.RAWVolumeData), BrickCoord);
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

	uint32_t AtlasBrickIndex = 0;
	for (uint32_t BrickIndex = 0; BrickIndex < BrickNum; ++BrickIndex)
	{
		if (BrickToAtlasBrick[BrickIndex] == CoordType(kInvalidCoordValue))
			continue;

		CoordType AtlasBrickCoord;
		AtlasBrickCoord.z = AtlasBrickIndex / BrickYxX;
		uint32_t Tmp = AtlasBrickIndex - AtlasBrickCoord.z * BrickYxX;
		AtlasBrickCoord.y = Tmp / VDBParams.BrickPerVolume.x;
		AtlasBrickCoord.x = Tmp - AtlasBrickCoord.y * VDBParams.BrickPerVolume.x;

		BrickToAtlasBrick[BrickIndex] = AtlasBrickCoord;

		++AtlasBrickIndex;
	}
	ValidBrickNum = AtlasBrickIndex;

	if (ValidBrickNum == 0)
		return;
	if (!resizeAtlas())
	{
		ValidBrickNum = 0;
		return;
	}

	uint32_t AtlasBrickNum =
		static_cast<uint32_t>(BrickPerAtlas.z) * BrickPerAtlas.y * BrickPerAtlas.x;
	AtlasBrickToBrick.assign(AtlasBrickNum, CoordType(kInvalidCoordValue));
	dAtlasBrickToBrick.resize(AtlasBrickNum);

	AtlasBrickIndex = 0;
	for (uint32_t BrickIndex = 0; BrickIndex < BrickNum; ++BrickIndex)
	{
		if (BrickToAtlasBrick[BrickIndex] == CoordType(kInvalidCoordValue))
			continue;

		CoordType BrickCoord;
		BrickCoord.z = BrickIndex / BrickYxX;
		uint32_t Tmp = BrickIndex - BrickCoord.z * BrickYxX;
		BrickCoord.y = Tmp / VDBParams.BrickPerVolume.x;
		BrickCoord.x = Tmp - BrickCoord.y * VDBParams.BrickPerVolume.x;

		AtlasBrickToBrick[AtlasBrickIndex] = BrickCoord;

		++AtlasBrickIndex;
	}

#ifdef ENABLE_CUDA_DEBUG_IN_CPU
	{
		std::string DebugMsg = "CUDA Debug Brick <-> AtlasBrick:\n\t";
		for (uint32_t BrickIndex = 0; BrickIndex < BrickNum; ++BrickIndex)
		{
			if (BrickToAtlasBrick[BrickIndex] == CoordType(kInvalidCoordValue))
				continue;

			DebugMsg += std::format(
				"b2a:{}->{}, ", BrickIndex, glm::to_string(BrickToAtlasBrick[BrickIndex]));
		}
		DebugMsg += "\n\t";

		for (uint32_t AtlasBrickIndex = 0; AtlasBrickIndex < AtlasBrickNum; ++AtlasBrickIndex)
		{
			if (AtlasBrickToBrick[AtlasBrickIndex] == CoordType(kInvalidCoordValue))
				continue;

			DebugMsg += std::format("a2b:{}->{}, ", AtlasBrickIndex,
				glm::to_string(AtlasBrickToBrick[AtlasBrickIndex]));
		}
		DebugMsg += "\n";

		std::cout << DebugMsg;
	}
#endif

	updateAtlas();
	CUDA_CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(dAtlasBrickToBrick.data()),
		AtlasBrickToBrick.data(), sizeof(CoordType) * AtlasBrickToBrick.size(),
		cudaMemcpyHostToDevice, AtlasStream));
	CUDA_CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(dBrickToAtlasBrick.data()),
		BrickToAtlasBrick.data(), sizeof(CoordType) * BrickToAtlasBrick.size(),
		cudaMemcpyHostToDevice, AtlasStream));

	CUDA_CHECK(cudaStreamSynchronize(AtlasStream));
}

bool DepthBoxVDB::VolData::VDBBuilder::resizeAtlas()
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
			return false;
		}

		if (AtlasArray && [&]() {
				auto Extent = AtlasArray->GetExtent();
				return Extent.width == NeededVoxelPerAtlas.x
					&& Extent.height == NeededVoxelPerAtlas.y
					&& Extent.depth == NeededVoxelPerAtlas.z;
			}())
			return true;
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

	return true;
}

void DepthBoxVDB::VolData::VDBBuilder::updateAtlas()
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
		uint32_t Tmp = AtlasBrickIndex - AtlasBrickCoord.z * BrickYxX;
		AtlasBrickCoord.y = Tmp / VDBParams.BrickPerVolume.x;
		AtlasBrickCoord.x = Tmp - AtlasBrickCoord.y * VDBParams.BrickPerVolume.x;

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

		CUDA_CHECK(cudaMemcpy3DAsync(&MemCpyParams, AtlasStream));
	};

	for (uint32_t AtlasBrickIndex = 0; AtlasBrickIndex < AtlasBrickToBrick.size();
		 ++AtlasBrickIndex)
	{
		if (AtlasBrickToBrick[AtlasBrickIndex] == CoordType(kInvalidCoordValue))
			continue;

		switch (VDBParams.VoxelType)
		{
			case EVoxelType::UInt8:
				Transfer(reinterpret_cast<uint8_t*>(BrickedData.data()), AtlasBrickIndex);
				break;
			case EVoxelType::UInt16:
				Transfer(reinterpret_cast<uint16_t*>(BrickedData.data()), AtlasBrickIndex);
				break;
			case EVoxelType::Float32:
				Transfer(reinterpret_cast<float*>(BrickedData.data()), AtlasBrickIndex);
				break;
			default:
				assert(false);
		}
	}
}
