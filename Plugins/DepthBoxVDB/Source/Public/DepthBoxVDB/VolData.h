#ifndef PUBLIC_DEPTHBOXVDB_VOLDATA_H
#define PUBLIC_DEPTHBOXVDB_VOLDATA_H

#include <numeric>
#include <memory>
#include <type_traits>

#include <array>

#include <glm/glm.hpp>

#include <DepthBoxVDB/Util.h>
#include <CUDA/Util.h>

namespace DepthBoxVDB
{
	namespace VolData
	{
		using CoordValueType = int32_t;
		using CoordType = glm::vec<3, CoordValueType>;

		static constexpr CoordValueType kInvalidCoordValue =
			std::numeric_limits<CoordValueType>::max();

		enum class EVoxelType : uint8_t
		{
			None = 0,
			UInt8,
			UInt16,
			Float32,
			MAX
		};

		inline size_t SizeOfVoxelType(EVoxelType VoxelType)
		{
			switch (VoxelType)
			{
				case EVoxelType::UInt8:
					return sizeof(uint8_t);
				case EVoxelType::UInt16:
					return sizeof(uint16_t);
				case EVoxelType::Float32:
					return sizeof(float);
				default:
					assert(false);
					return 0;
			}
		}

		inline cudaChannelFormatDesc CUDAChannelDescOfVoxelType(EVoxelType VoxelType)
		{
			switch (VoxelType)
			{
				case EVoxelType::UInt8:
					return cudaCreateChannelDesc<uint8_t>();
				case EVoxelType::Float32:
					return cudaCreateChannelDesc<float>();
					break;
				default:
					assert(false);
					return {};
			}
		}

		struct CUDA_ALIGN VDBParameters
		{
			static constexpr int32_t kMaxLevelNum = 3;

			EVoxelType VoxelType;
			int32_t	   RootLevel;
			int32_t	   ApronWidth;
			int32_t	   ApronAndDepthWidth;
			int32_t	   LogChildPerLevels[kMaxLevelNum];
			int32_t	   ChildPerLevels[kMaxLevelNum];
			int32_t	   ChildCoverVoxelPerLevels[kMaxLevelNum];
			int32_t	   DepthCoordValueInAtlasBrick[2];
			int32_t	   VoxelPerAtlasBrick;
			CoordType  VoxelPerVolume;
			CoordType  BrickPerVolume;
		};

		class IVDBBuilder : Noncopyable
		{
		public:
			struct CreateParameters
			{
			};
			static std::shared_ptr<IVDBBuilder> Create(const CreateParameters& Params);
			virtual ~IVDBBuilder() {}

			struct FullBuildParameters
			{
				uint8_t*		 RAWVolumeData;
				const glm::vec2* EmptyScalarRanges;
				uint32_t		 EmptyScalarRangeNum;
				uint32_t		 MaxAllowedGPUMemoryInGB;
				VDBParameters	 VDBParams;
			};
			virtual void FullBuild(const FullBuildParameters& Params) = 0;

			struct UpdateDepthBoxParameters
			{
				const glm::vec2* EmptyScalarRanges;
				uint32_t		 EmptyScalarRangeNum;
			};
			virtual void UpdateDepthBoxAsync(const UpdateDepthBoxParameters& Params) = 0;
		};

	} // namespace VolData
} // namespace DepthBoxVDB

#endif
