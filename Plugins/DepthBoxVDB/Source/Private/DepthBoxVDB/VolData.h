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
				VDBNode Ret{ CoordType(kInvalidCoordValue), CoordType(kInvalidCoordValue),
					std::numeric_limits<uint64_t>::max() };
				return Ret;
			}
		};

		struct CUDA_ALIGN VDBData
		{
			static constexpr uint32_t kInvalidChild = std::numeric_limits<uint32_t>::max();

			VDBNode*  NodePerLevels[VDBParameters::kMaxLevelNum] = { nullptr };
			uint32_t* ChildPerLevels[VDBParameters::kMaxLevelNum - 1] = { nullptr };

			cudaSurfaceObject_t AtlasSurface = 0;
			cudaTextureObject_t AtlasTexture = 0;

			VDBParameters VDBParams;

			__host__ __device__ VDBNode& Node(int32_t Level, uint32_t Index) const
			{
				return NodePerLevels[Level][Index];
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

			__host__ __device__ CoordType MapCoord(
				int32_t DstLevel, int32_t SrcLevel, const CoordType& SrcCoord)
			{
				return SrcCoord * VDBParams.ChildCoverVoxelPerLevels[SrcLevel + 1]
					/ VDBParams.ChildCoverVoxelPerLevels[DstLevel + 1];
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

		class VDB : public IVDB
		{
		public:
			VDB(const CreateParameters& Params);
			~VDB();

			void FullBuild(const FullBuildParameters& Params) override;
			void UpdateDepthBoxAsync(const UpdateDepthBoxParameters& Params) override;

			VDBData* GetDeviceData() const;

			// Declare Private functions in Public scope to use CUDA Lambda
			template <typename VoxelType>
			void updateDepthBoxAsync(const UpdateDepthBoxParameters& Params);
			void relayoutRAWVolume(const FullBuildParameters& Params);
			bool resizeAtlas();
			void updateAtlas();

		private:
			uint32_t	 ValidBrickNum = 0;
			uint32_t	 MaxAllowedGPUMemoryInGB;
			CoordType	 BrickPerAtlas;
			cudaStream_t AtlasStream = 0;
			cudaStream_t NodeStream = 0;

			VDBParameters VDBParams;

			std::shared_ptr<CUDA::Array>   AtlasArray;
			std::unique_ptr<CUDA::Texture> AtlasTexture;
			std::unique_ptr<CUDA::Surface> AtlasSurface;

			std::vector<uint8_t> BrickedData;

			std::vector<CoordType>			 AtlasBrickToBrick;
			std::vector<CoordType>			 BrickToAtlasBrick;
			thrust::device_vector<CoordType> dAtlasBrickToBrick;
			thrust::device_vector<CoordType> dBrickToAtlasBrick;

			VDBData* dData = nullptr;

			std::array<thrust::device_vector<VDBNode>, VDBParameters::kMaxLevelNum> dNodePerLevels;
			std::array<thrust::device_vector<uint32_t>, VDBParameters::kMaxLevelNum>
				dChildPerLevels;

			friend class VolRenderer::VDBRenderer;
		};

	} // namespace VolData
} // namespace DepthBoxVDB

#endif
