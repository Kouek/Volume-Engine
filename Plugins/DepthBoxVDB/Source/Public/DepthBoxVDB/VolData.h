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

		static constexpr CoordValueType InvalidCoordValue =
			std::numeric_limits<CoordValueType>::max();

		enum class EVoxelType : uint8_t
		{
			None = 0,
			UInt8,
			Float32,
			MAX
		};

		inline size_t SizeOfVoxelType(EVoxelType VoxelType)
		{
			switch (VoxelType)
			{
				case EVoxelType::UInt8:
					return sizeof(uint8_t);
				case EVoxelType::Float32:
					return sizeof(float);
				default:
					return 0;
			}
		}

		inline cudaChannelFormatDesc CUDAChannelDescOfVoxelType(EVoxelType VoxelType)
		{
			cudaChannelFormatDesc ChannelDesc{};
			switch (VoxelType)
			{
				case EVoxelType::UInt8:
					ChannelDesc = cudaCreateChannelDesc<uint8_t>();
					break;
				case EVoxelType::Float32:
					ChannelDesc = cudaCreateChannelDesc<float>();
					break;
			}

			return ChannelDesc;
		}

		struct CUDA_ALIGN VDBParameters
		{
			static constexpr int32_t MaxLevelNum = 3;
			static constexpr int32_t MaxLogChildPerLevel = 9;

			EVoxelType VoxelType;
			int32_t	   RootLevel;
			int32_t	   ApronWidth;
			int32_t	   ApronAndDepthWidth;
			int32_t	   LogChildPerLevels[MaxLevelNum];
			int32_t	   ChildPerLevels[MaxLevelNum];
			int32_t	   ChildCoverVoxelPerLevels[MaxLevelNum];
			int32_t	   DepthCoordValueInAtlasBrick[2];
			int32_t	   VoxelPerAtlasBrick;
			CoordType  VoxelPerVolume;
			CoordType  BrickPerVolume;
		};

		class IVDBDataProvider : Noncopyable
		{
		public:
			struct CreateParameters
			{
				uint8_t*	  RAWVolumeData;
				glm::vec2*	  EmptyScalarRanges;
				uint32_t	  EmptyScalarRangeNum;
				uint32_t	  MaxAllowedGPUMemoryInGB;
				VDBParameters VDBParams;
			};
			static std::shared_ptr<IVDBDataProvider> Create(const CreateParameters& Params);
			virtual ~IVDBDataProvider() {}
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
				uint32_t						  EmptyScalarRangeNum;
				glm::vec2*						  EmptyScalarRanges;
				std::shared_ptr<IVDBDataProvider> Provider;
			};
			virtual void FullBuild(const FullBuildParameters& Params) = 0;
		};

	} // namespace VolData
} // namespace DepthBoxVDB

#endif
