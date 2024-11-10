#ifndef PRIVATE_DEPTHBOXVDB_VOLDATA_H
#define PRIVATE_DEPTHBOXVDB_VOLDATA_H

#include <DepthBoxVDB/VolData.h>

#include <memory>

#include <array>
#include <unordered_map>

#include <thrust/device_vector.h>

#include <CUDA/Types.h>

#include "Util.h"

namespace DepthBoxVDB
{
	namespace VolRenderer
	{
		class VDBRenderer;
	}

	namespace VolData
	{
		struct CUDA_ALIGN VDBNode
		{
			CoordType Coord;
			CoordType CoordInAtlas;
			uint64_t  ChildListOffset;

			__host__ __device__ static VDBNode CreateInvalid()
			{
				VDBNode Ret{ CoordType(InvalidCoordValue), CoordType(InvalidCoordValue),
					std::numeric_limits<uint64_t>::max() };
				return Ret;
			}
		};

		struct CUDA_ALIGN VDBData
		{
			static constexpr uint32_t InvalidChild = std::numeric_limits<uint32_t>::max();

			VDBNode*  NodePerLevels[VDBParameters::MaxLevelNum] = { nullptr };
			uint32_t* ChildPerLevels[VDBParameters::MaxLevelNum - 1] = { nullptr };

			cudaSurfaceObject_t AtlasSurface = 0;
			cudaTextureObject_t AtlasTexture = 0;

			VDBParameters VDBParams;

			__host__ __device__ uint32_t Index(int32_t Level, const CoordType& Coord) const
			{
				if (Level == VDBParams.RootLevel)
					return 0;

				int32_t ChildCurrLev = VDBParams.ChildPerLevels[Level + 1];
				return static_cast<uint32_t>(Coord.z) * ChildCurrLev * ChildCurrLev
					+ Coord.y * ChildCurrLev + Coord.x;
			}

			__host__ __device__ VDBNode& Node(int32_t Level, uint32_t Index) const
			{
				return NodePerLevels[Level][Index];
			}
			__host__ __device__ VDBNode& Node(int32_t Level, const CoordType& Coord) const
			{
				return Node(Level, Index(Level, Coord));
			}
			__host__ __device__ uint32_t& Child(
				int32_t ParentLevel, uint32_t ChildIndexInParent, const VDBNode& Parent) const
			{
				return ChildPerLevels[ParentLevel - 1][Parent.ChildListOffset + ChildIndexInParent];
			}
			__host__ __device__ uint32_t& Child(int32_t ParentLevel,
				const CoordType& ChildCoordInParent, const VDBNode& Parent) const
			{
				return Child(
					ParentLevel, ChildIndexInParent(ParentLevel, ChildCoordInParent), Parent);
			}

			__host__ __device__ CoordType CoordInParentLevel(
				int32_t ParentLevel, const CoordType& Coord) const
			{
				return Coord * VDBParams.ChildCoverVoxelPerLevels[ParentLevel - 1]
					/ VDBParams.ChildCoverVoxelPerLevels[ParentLevel]
					* VDBParams.ChildCoverVoxelPerLevels[ParentLevel - 1];
			}
			__host__ __device__ CoordType ChildCoordInParent(
				int32_t ParentLevel, const CoordType& Coord) const
			{
				CoordType VoxelPositionMin =
					Coord * VDBParams.ChildCoverVoxelPerLevels[ParentLevel - 1];
				CoordType VoxelPositionMinParent = VoxelPositionMin
					/ VDBParams.ChildCoverVoxelPerLevels[ParentLevel]
					* VDBParams.ChildCoverVoxelPerLevels[ParentLevel - 1];
				return (VoxelPositionMin - VoxelPositionMinParent)
					/ VDBParams.ChildCoverVoxelPerLevels[ParentLevel - 1];
			}
			__host__ __device__ uint32_t ChildIndexInParent(
				int32_t ParentLevel, const CoordType& ChildCoordInParent) const
			{
				int32_t LogChild = VDBParams.LogChildPerLevels[ParentLevel];
				return (static_cast<uint32_t>(ChildCoordInParent.z) << (LogChild << 1))
					| (static_cast<uint32_t>(ChildCoordInParent.y) << LogChild)
					| static_cast<uint32_t>(ChildCoordInParent.x);
			}
		};

		class VDBBuilder;

		class VDBProvider : public IVDBDataProvider
		{
		public:
			VDBProvider(const CreateParameters& Params);
			~VDBProvider();

		private:
			void relayoutRAWVolume(const CreateParameters& Params);
			void resizeAtlas();
			void updateAtlas();

		private:
			uint32_t	 ValidBrickNum = 0;
			uint32_t	 MaxAllowedGPUMemoryInGB;
			CoordType	 BrickPerAtlas;
			cudaStream_t Stream = 0;

			VDBParameters VDBParams;

			std::shared_ptr<CUDA::Array>   AtlasArray;
			std::unique_ptr<CUDA::Texture> AtlasTexture;
			std::unique_ptr<CUDA::Surface> AtlasSurface;

			std::vector<uint8_t> BrickedData;

			std::vector<CoordType>			 AtlasBrickToBrick;
			std::vector<CoordType>			 BrickToAtlasBrick;
			thrust::device_vector<CoordType> dAtlasBrickToBrick;
			thrust::device_vector<CoordType> dBrickToAtlasBrick;

			friend class VDBBuilder;
			friend class VolRenderer::VDBRenderer;
		};

		class VDBBuilder : public IVDBBuilder
		{
		public:
			VDBBuilder(const CreateParameters& Params);
			~VDBBuilder();

			void FullBuild(const FullBuildParameters& Params) override;

			VDBData* GetDeviceData() const;

		private:
			cudaStream_t Stream = 0;
			VDBData*	 dData = nullptr;

			std::shared_ptr<VDBProvider> Provider;

			std::array<thrust::device_vector<VDBNode>, VDBParameters::MaxLevelNum>	dNodePerLevels;
			std::array<thrust::device_vector<uint32_t>, VDBParameters::MaxLevelNum> dChildPerLevels;

			friend class VolRenderer::VDBRenderer;
		};

	} // namespace VolData
} // namespace DepthBoxVDB

#endif
