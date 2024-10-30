#include "VolData.h"

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
	return std::make_unique<VDBDataProvider>(Params);
}

std::unique_ptr<DepthBoxVDB::VolData::IVDBBuilder> DepthBoxVDB::VolData::IVDBBuilder::Create(
	const CreateParameters& Params)
{
	return std::make_unique<VDBBuilder>(Params);
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

	__host__ __device__ constexpr static BrickSortKey Invalid()
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

DepthBoxVDB::VolData::VDBDataProvider::VDBDataProvider(const CreateParameters& Params)
{
	int DeviceNum = 0;
	CUDA_CHECK(cudaGetDeviceCount(&DeviceNum));
	assert(DeviceNum > 0);

	cudaDeviceProp Prop;
	CUDA_CHECK(cudaGetDeviceProperties(&Prop, 0));

	CUDA_CHECK(cudaStreamCreate(&Stream));
}

DepthBoxVDB::VolData::VDBDataProvider::~VDBDataProvider() {}

void DepthBoxVDB::VolData::VDBDataProvider::TransferRAWVolumeToAtlas(
	const TransferRAWVolumeToAtlasParameters& Params)
{
	auto& VDBParams = Params.VDBParams;

	resizeAtlasArray(Params.VDBParams);
	if (!AtlasArray->IsComplete())
	{
		std::cerr << "Invalid AtlasArray\n";
		return;
	}

	// Temporarily store the whole Volume in Atlas
	{
		cudaMemcpy3DParms MemCpyParams{};
		MemCpyParams.srcPtr = make_cudaPitchedPtr(Params.RAWVolumeData,
			SizeOfVoxelType(VDBParams.VoxelType) * Params.VoxelPerVolume.x, Params.VoxelPerVolume.x,
			Params.VoxelPerVolume.y);
		MemCpyParams.extent = make_cudaExtent(
			Params.VoxelPerVolume.x, Params.VoxelPerVolume.y, Params.VoxelPerVolume.z);
		MemCpyParams.dstArray = AtlasArray->Get();
		MemCpyParams.kind = cudaMemcpyHostToDevice;
		CUDA_CHECK(cudaMemcpy3D(&MemCpyParams));

		AtlasTexture = std::make_unique<CUDA::Texture>(AtlasArray);
		AtlasSurface = std::make_unique<CUDA::Surface>(AtlasArray);
	}

	// Shift Atlas layout to contain Apron and Depth
	{
		auto ShiftKernel = [AtlasSurface = AtlasSurface->Get(), VDBParams] __device__(
							   int32_t Axis, const CoordType& BrickCoord) {
			if (BrickCoord.x >= VDBParams.BrickPerVolume.x
				|| BrickCoord.y >= VDBParams.BrickPerVolume.y
				|| BrickCoord.z >= VDBParams.BrickPerVolume.z)
				return;

			int32_t VoxelPerBrick = VDBParams.ChildPerLevels[0];
			int32_t OffsetOnAxis =
				BrickCoord[Axis] * VDBParams.ApronAndDepthWidth + VDBParams.ApronAndDepthWidth;
			CoordType VoxelPositionMin = BrickCoord
					* CoordType(VDBParams.VoxelPerAtlasBrick,
						Axis >= 1 ? VDBParams.VoxelPerAtlasBrick : VoxelPerBrick,
						Axis >= 2 ? VDBParams.VoxelPerAtlasBrick : VoxelPerBrick)
				+ CoordType(VDBParams.ApronAndDepthWidth,
					Axis >= 1 ? VDBParams.ApronAndDepthWidth : 0,
					Axis >= 2 ? VDBParams.ApronAndDepthWidth : 0);

			auto Shift = [&]<typename T>(CoordValueType X, CoordValueType Y, CoordValueType Z, T*) {
				T Scalar =
					surf3Dread<T>(AtlasSurface, sizeof(T) * (Axis == 0 ? X - OffsetOnAxis : X),
						Axis == 1 ? Y - OffsetOnAxis : Y, Axis == 2 ? Z - OffsetOnAxis : Z);
				surf3Dwrite(Scalar, AtlasSurface, sizeof(T) * X, Y, Z);
			};

			for (CoordValueType dZ = VoxelPerBrick - 1; dZ >= 0; --dZ)
				for (CoordValueType dY = VoxelPerBrick - 1; dY >= 0; --dY)
					for (CoordValueType dX = VoxelPerBrick - 1; dX >= 0; --dX)
					{
						switch (VDBParams.VoxelType)
						{
							case EVoxelType::UInt8:
								Shift(VoxelPositionMin.x + dX, VoxelPositionMin.y + dY,
									VoxelPositionMin.z + dZ, (uint8_t*)nullptr);
								break;
							case EVoxelType::Float32:
								Shift(VoxelPositionMin.x + dX, VoxelPositionMin.y + dY,
									VoxelPositionMin.z + dZ, (float*)nullptr);
								break;
						}
					}
		};

		for (int32_t Axis = 0; Axis < 3; ++Axis)
		{
			dim3 ThreadPerBlock3D;
			ThreadPerBlock3D.x = Axis == 0 ? 1 : CUDA::ThreadPerBlockX2D;
			ThreadPerBlock3D.y = Axis == 1 ? 1 : CUDA::ThreadPerBlockX2D;
			ThreadPerBlock3D.z = Axis == 2 ? 1 : CUDA::ThreadPerBlockX2D;

			dim3 BlockPerGrid;
			BlockPerGrid.x = Axis == 0
				? 1
				: (VDBParams.BrickPerVolume.x + ThreadPerBlock3D.x - 1) / ThreadPerBlock3D.x;
			BlockPerGrid.y = Axis == 1
				? 1
				: (VDBParams.BrickPerVolume.y + ThreadPerBlock3D.y - 1) / ThreadPerBlock3D.y;
			BlockPerGrid.z = Axis == 2
				? 1
				: (VDBParams.BrickPerVolume.z + ThreadPerBlock3D.z - 1) / ThreadPerBlock3D.z;

			for (int32_t BrickStart = VDBParams.BrickPerVolume[Axis] - 1; BrickStart >= 0;
				 --BrickStart)
			{
				CUDA::ParallelFor(
					BlockPerGrid, ThreadPerBlock3D,
					[Axis, BrickStart, ShiftKernel] __device__(
						const glm::uvec3& DispatchThreadIdx) {
						ShiftKernel(Axis,
							CoordType(DispatchThreadIdx.x + (Axis == 0 ? BrickStart : 0),
								DispatchThreadIdx.y + (Axis == 1 ? BrickStart : 0),
								DispatchThreadIdx.z + (Axis == 2 ? BrickStart : 0)));
					},
					Stream);
			}
		}
	}

#ifdef ENABLE_CUDA_DEBUG_IN_CPU
	// Read back the last brick and check
	assert(VDBParams.VoxelType == EVoxelType::UInt8);
	CUDA_CHECK(cudaStreamSynchronize(Stream));
	std::vector<uint8_t> Readbacks(
		VDBParams.ChildPerLevels[0] * VDBParams.ChildPerLevels[0] * VDBParams.ChildPerLevels[0]);
	{
		CoordType VoxelPositionMin = (VDBParams.BrickPerVolume - 1) * VDBParams.VoxelPerAtlasBrick
			+ CoordType(VDBParams.ApronAndDepthWidth, VDBParams.ApronAndDepthWidth,
				VDBParams.ApronAndDepthWidth);

		cudaMemcpy3DParms MemCpyParams{};
		MemCpyParams.srcArray = AtlasArray->Get();
		MemCpyParams.srcPos.x = VoxelPositionMin.x;
		MemCpyParams.srcPos.y = VoxelPositionMin.y;
		MemCpyParams.srcPos.z = VoxelPositionMin.z;
		MemCpyParams.extent = make_cudaExtent(
			VDBParams.ChildPerLevels[0], VDBParams.ChildPerLevels[0], VDBParams.ChildPerLevels[0]);
		MemCpyParams.dstPtr =
			make_cudaPitchedPtr(Readbacks.data(), sizeof(uint8_t) * VDBParams.ChildPerLevels[0],
				VDBParams.ChildPerLevels[0], VDBParams.ChildPerLevels[0]);
		MemCpyParams.kind = cudaMemcpyDeviceToHost;

		CUDA_CHECK(cudaMemcpy3D(&MemCpyParams));
	}
	{
		CoordType	VoxelPositionMin = (VDBParams.BrickPerVolume - 1) * VDBParams.ChildPerLevels[0];
		std::string DebugMsg;
		for (CoordValueType dZ = 0; dZ < VDBParams.ChildPerLevels[0]; ++dZ)
			for (CoordValueType dY = 0; dY < VDBParams.ChildPerLevels[0]; ++dY)
				for (CoordValueType dX = 0; dX < VDBParams.ChildPerLevels[0]; ++dX)
				{
					CoordType VoxelPosition = glm::min(
						VoxelPositionMin + CoordType(dX, dY, dZ), Params.VoxelPerVolume - 1);

					uint8_t ScalarOrigin = Params.RAWVolumeData[VoxelPosition.z
							* Params.VoxelPerVolume.y * Params.VoxelPerVolume.x
						+ VoxelPosition.y * Params.VoxelPerVolume.x + VoxelPosition.x];
					uint8_t ScalarReadback =
						Readbacks[dZ * VDBParams.ChildPerLevels[0] * VDBParams.ChildPerLevels[0]
							+ dY * VDBParams.ChildPerLevels[0] + dX];

					if (ScalarOrigin != ScalarReadback)
					{
						DebugMsg += std::format("({},{},{}), ", VoxelPositionMin.x + dX,
							VoxelPositionMin.y + dY, VoxelPositionMin.z + dZ);
					}
				}
		std::cout << DebugMsg;
	}
#endif
}

void DepthBoxVDB::VolData::VDBDataProvider::resizeAtlasArray(const VDBParameters& Params)
{
	if (AtlasArray && [&]() {
			auto Extent = AtlasArray->GetExtent();
			return Extent.width == Params.InitialVoxelPerAtlas.x
				&& Extent.height == Params.InitialVoxelPerAtlas.y
				&& Extent.depth == Params.InitialVoxelPerAtlas.z;
		}())
		return;

	cudaChannelFormatDesc ChannelDesc;
	switch (Params.VoxelType)
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
	AtlasArray = std::make_shared<CUDA::Array>(ChannelDesc, Params.InitialVoxelPerAtlas);
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
template <typename T>
inline void DebugInCPU(const thrust::device_vector<T>& dVector, const char* Name)
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
	if (cudaStreamQuery(Stream) == cudaErrorNotReady)
	{
		CUDA_CHECK(cudaStreamSynchronize(Stream));
	}

	Provider = std::static_pointer_cast<VDBDataProvider>(Params.Provider);
	if (cudaStreamQuery(Provider->Stream) == cudaErrorNotReady)
	{
		CUDA_CHECK(cudaStreamSynchronize(Provider->Stream));
	}

	if (!Provider->AtlasSurface)
	{
		std::cerr << "Invalid Provider\n";
		return;
	}

	auto&	 VDBParams = Params.VDBParams;
	uint32_t BrickYxX = VDBParams.BrickPerVolume.x * VDBParams.BrickPerVolume.y;
	uint32_t BrickNum = BrickYxX * VDBParams.BrickPerVolume.z;
	uint32_t VoxelNumPerBrick =
		VDBParams.ChildPerLevels[0] * VDBParams.ChildPerLevels[0] * VDBParams.ChildPerLevels[0];

	const dim3 ThreadPerBlock3D = { CUDA::ThreadPerBlockX3D, CUDA::ThreadPerBlockY3D,
		CUDA::ThreadPerBlockZ3D };

	// 1. Check empty Voxels in each Brick, find empty Bricks
	// 2. Assign Brick Sort Keys to non-empty Brick
	// 3. Sort Brick Sort Keys
	thrust::device_vector<BrickSortKey> dBrickSortKeys(
		(VDBParams.RootLevel + 1) * BrickNum, BrickSortKey::Invalid());
	{
		thrust::device_vector<glm::vec2> dEmptyScalarRanges(Params.EmptyScalarRangeNum);
		CUDA_CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(dEmptyScalarRanges.data()),
			Params.EmptyScalarRanges, sizeof(glm::vec2) * Params.EmptyScalarRangeNum,
			cudaMemcpyHostToDevice, Stream));

		auto AssignBrickKeysKernel =
			[BrickSortKeys = thrust::raw_pointer_cast(dBrickSortKeys.data()),
				EmptyScalarRanges = thrust::raw_pointer_cast(dEmptyScalarRanges.data()),
				AtlasSurface = Provider->AtlasSurface->Get(), VoxelNumPerBrick, BrickYxX, BrickNum,
				EmptyScalarRangeNum = Params.EmptyScalarRangeNum,
				VDBParams] __device__(const glm::uvec3& BrickCoord) {
				if (BrickCoord.x >= VDBParams.BrickPerVolume.x
					|| BrickCoord.y >= VDBParams.BrickPerVolume.y
					|| BrickCoord.z >= VDBParams.BrickPerVolume.z)
					return;

				CoordType VoxelPositionMin = static_cast<CoordType>(BrickCoord)
						* static_cast<CoordValueType>(VDBParams.VoxelPerAtlasBrick)
					+ static_cast<CoordValueType>(VDBParams.ApronAndDepthWidth);

				uint32_t EmptyVoxelNum = 0;
				for (CoordValueType dZ = 0; dZ < VDBParams.ChildPerLevels[0]; ++dZ)
					for (CoordValueType dY = 0; dY < VDBParams.ChildPerLevels[0]; ++dY)
						for (CoordValueType dX = 0; dX < VDBParams.ChildPerLevels[0]; ++dX)
						{
							float Scalar;
							switch (VDBParams.VoxelType)
							{
								case EVoxelType::UInt8:
									Scalar = surf3Dread<uint8_t>(AtlasSurface,
										sizeof(uint8_t) * (VoxelPositionMin.x + dX),
										VoxelPositionMin.y + dY, VoxelPositionMin.z + dZ);
									break;
								case EVoxelType::Float32:
									Scalar = surf3Dread<float>(AtlasSurface,
										sizeof(float) * (VoxelPositionMin.x + dX),
										VoxelPositionMin.y + dY, VoxelPositionMin.z + dZ);
									break;
							}

							for (uint32_t RngIdx = 0; RngIdx < EmptyScalarRangeNum; ++RngIdx)
							{
								glm::vec2 Range = EmptyScalarRanges[RngIdx];
								if (Range[0] <= Scalar && Scalar <= Range[1])
								{
									++EmptyVoxelNum;
								}
							}
						}

				if (EmptyVoxelNum == VoxelNumPerBrick)
					return;

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

		dim3 BlockPerGrid;
		BlockPerGrid.x = (VDBParams.BrickPerVolume.x + ThreadPerBlock3D.x - 1) / ThreadPerBlock3D.x;
		BlockPerGrid.y = (VDBParams.BrickPerVolume.y + ThreadPerBlock3D.y - 1) / ThreadPerBlock3D.y;
		BlockPerGrid.z = (VDBParams.BrickPerVolume.z + ThreadPerBlock3D.z - 1) / ThreadPerBlock3D.z;
		CUDA::ParallelFor(BlockPerGrid, ThreadPerBlock3D, AssignBrickKeysKernel, Stream);

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

			dNodePerLevels[Lev].resize(NumCurrLev, VDBNode::Invalid());

			if (Lev > 0)
			{
				uint64_t ChildCurrLev = VDBParams.ChildPerLevels[Lev];
				dChildPerLevels[Lev - 1].resize(
					dNodePerLevels[Lev].size() * ChildCurrLev * ChildCurrLev * ChildCurrLev);
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
		auto AssignNodePoolsKernel = [BrickYxX, Data = dData,
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
				Node.CoordInAtlas.z = NodeIndex / BrickYxX;
				uint32_t Tmp = Node.CoordInAtlas.z * BrickYxX;
				Node.CoordInAtlas.y = (NodeIndex - Tmp) / VDBParams.BrickPerVolume.x;
				Node.CoordInAtlas.x =
					NodeIndex - Tmp - Node.CoordInAtlas.y * VDBParams.BrickPerVolume.x;

				Node.CoordInAtlas *= VDBParams.VoxelPerAtlasBrick;
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
			thrust::for_each(thrust::cuda::par.on(Stream),
				thrust::make_counting_iterator(uint32_t(0)),
				thrust::make_counting_iterator(static_cast<uint32_t>(dNodePerLevels[Lev].size())),
				[Lev, NodeIndexStart, AssignNodePoolsKernel] __device__(
					uint32_t NodeIndex) { AssignNodePoolsKernel(Lev, NodeIndexStart, NodeIndex); });
			NodeIndexStart += dNodePerLevels[Lev].size();
		}
		{
			VDBNode Root = VDBNode::Invalid();
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
