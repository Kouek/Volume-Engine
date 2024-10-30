#include "VolData.h"

#include <thrust/binary_search.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/system/cuda/execution_policy.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <glm/gtx/string_cast.hpp>

#include <CUDA/Algorithm.h>

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

DepthBoxVDB::VolData::VDBBuilder::VDBBuilder(const CreateParameters& Params)
{
	int DeviceNum = 0;
	CUDA_CHECK(cudaGetDeviceCount(&DeviceNum));
	assert(DeviceNum > 0);

	cudaDeviceProp Prop;
	CUDA_CHECK(cudaGetDeviceProperties(&Prop, 0));

	CUDA_CHECK(cudaStreamCreate(&Stream));
}

#define CUDA_DEBUG_IN_CPU

#ifdef CUDA_DEBUG_IN_CPU
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

	#define DEBUG_IN_CPU(X) DebugInCPU(X, #X)

#endif

void DepthBoxVDB::VolData::VDBBuilder::FullBuild(const FullBuildParameters& Params)
{
	if (cudaStreamQuery(Stream) == cudaErrorNotReady)
	{
		CUDA_CHECK(cudaStreamSynchronize(Stream));
	}

	auto&	 VDBParams = Params.VDBParams;
	uint32_t BrickYxX = VDBParams.BrickPerVolume.x * VDBParams.BrickPerVolume.y;
	uint32_t BrickNum = BrickYxX * VDBParams.BrickPerVolume.z;
	uint32_t VoxelNumPerBrick =
		VDBParams.ChildPerLevels[0] * VDBParams.ChildPerLevels[0] * VDBParams.ChildPerLevels[0];

	const dim3 ThreadPerBlock3D = { CUDA::ThreadPerBlockX3D, CUDA::ThreadPerBlockY3D,
		CUDA::ThreadPerBlockZ3D };

	resizeAtlasArray(Params);
	if (!AtlasArray->IsComplete())
		return;

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

		cudaTextureDesc TexDesc{};
		TexDesc.normalizedCoords = 0;
		TexDesc.filterMode = cudaFilterModePoint;
		TexDesc.addressMode[0] = TexDesc.addressMode[1] = TexDesc.addressMode[2] =
			cudaAddressModeBorder;
		TexDesc.readMode = cudaReadModeElementType;
		AtlasTexture = std::make_shared<CUDA::Texture>(AtlasArray, TexDesc);
	}

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
				VolumeTexture = AtlasTexture->Get(), VoxelNumPerBrick, BrickYxX, BrickNum,
				EmptyScalarRangeNum = Params.EmptyScalarRangeNum,
				VDBParams] __device__(const glm::uvec3& DispatchThreadIdx) {
				CoordType VoxelPositionMin = static_cast<CoordType>(DispatchThreadIdx)
					* static_cast<CoordValueType>(VDBParams.ChildPerLevels[0]);

				uint32_t EmptyVoxelNum = 0;
				for (CoordValueType dX = 0; dX < VDBParams.ChildPerLevels[0]; ++dX)
					for (CoordValueType dY = 0; dY < VDBParams.ChildPerLevels[0]; ++dY)
						for (CoordValueType dZ = 0; dZ < VDBParams.ChildPerLevels[0]; ++dZ)
						{
							float Scalar;
							switch (VDBParams.VoxelType)
							{
								case EVoxelType::UInt8:
									Scalar = tex3D<uint8_t>(VolumeTexture, VoxelPositionMin.x + dX,
										VoxelPositionMin.y + dY, VoxelPositionMin.z + dZ);
									break;
								case EVoxelType::Float32:
									Scalar = tex3D<float>(VolumeTexture, VoxelPositionMin.x + dX,
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
				Key.LevelPosition.X = DispatchThreadIdx.x;
				Key.LevelPosition.Y = DispatchThreadIdx.y;
				Key.LevelPosition.Z = DispatchThreadIdx.z;
				uint32_t FlatIdx = DispatchThreadIdx.z * BrickYxX
					+ DispatchThreadIdx.y * VDBParams.BrickPerVolume.x + DispatchThreadIdx.x;
				BrickSortKeys[FlatIdx] = Key;

				for (int32_t Lev = 1; Lev <= VDBParams.RootLevel; ++Lev)
				{
					Key.LevelPosition.Level = Lev;
					Key.LevelPosition.X /= VDBParams.ChildPerLevels[Lev];
					Key.LevelPosition.Y /= VDBParams.ChildPerLevels[Lev];
					Key.LevelPosition.Z /= VDBParams.ChildPerLevels[Lev];

					FlatIdx = Lev * BrickNum + DispatchThreadIdx.z * BrickYxX
						+ DispatchThreadIdx.y * VDBParams.BrickPerVolume.x + DispatchThreadIdx.x;
					BrickSortKeys[FlatIdx] = Key;
				}
			};

		dim3 BlockPerGrid;
		BlockPerGrid.x = (VDBParams.BrickPerVolume.x + ThreadPerBlock3D.x - 1) / ThreadPerBlock3D.x;
		BlockPerGrid.y = (VDBParams.BrickPerVolume.y + ThreadPerBlock3D.y - 1) / ThreadPerBlock3D.y;
		BlockPerGrid.z = (VDBParams.BrickPerVolume.z + ThreadPerBlock3D.z - 1) / ThreadPerBlock3D.z;
		CUDA::ParallelFor(BlockPerGrid, ThreadPerBlock3D, AssignBrickKeysKernel, Stream);

		thrust::sort(thrust::cuda::par.on(Stream), dBrickSortKeys.begin(), dBrickSortKeys.end());

#ifdef CUDA_DEBUG_IN_CPU
		CUDA_CHECK(cudaStreamSynchronize(Stream));
		DEBUG_IN_CPU(dEmptyScalarRanges);
		DEBUG_IN_CPU(dBrickSortKeys);
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

#ifdef CUDA_DEBUG_IN_CPU
		CUDA_CHECK(cudaStreamSynchronize(Stream));
		DEBUG_IN_CPU(dBrickSortKeys);
#endif
	}
}

void DepthBoxVDB::VolData::VDBBuilder::resizeAtlasArray(const FullBuildParameters& Params)
{
	if (AtlasArray && [&]() {
			auto Extent = AtlasArray->GetExtent();
			return Extent.width == Params.VDBParams.InitialVoxelPerAtlas.x
				&& Extent.height == Params.VDBParams.InitialVoxelPerAtlas.y
				&& Extent.depth == Params.VDBParams.InitialVoxelPerAtlas.z;
		}())
		return;

	cudaChannelFormatDesc ChannelDesc;
	switch (Params.VDBParams.VoxelType)
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
	AtlasArray = std::make_shared<CUDA::Array>(ChannelDesc, Params.VDBParams.InitialVoxelPerAtlas);
}
