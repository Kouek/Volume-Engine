#ifndef PUBLIC_DEPTHBOXVDB_VOLDATA_H
#define PUBLIC_DEPTHBOXVDB_VOLDATA_H

#include <numeric>
#include <memory>
#include <type_traits>

#include <array>

#include <glm/glm.hpp>

#include <CUDA/Util.h>

namespace DepthBoxVDB
{
	namespace VolData
	{
		using CoordValueType = int32_t;
		using CoordType = glm::vec<3, CoordValueType>;

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
			int32_t	   ApronWidth = 1;
			int32_t	   ApronAndDepthWidth;
			int32_t	   LogChildPerLevels[MaxLevelNum] = { 5, 4, 3 };
			int32_t	   ChildPerLevels[MaxLevelNum];
			int32_t	   ChildCoverVoxelPerLevels[MaxLevelNum];
			int32_t	   DepthPositionInAtlasBrick[2];
			bool	   bUseDepthBox = true;
			int32_t	   VoxelPerAtlasBrick;
			CoordType  BrickPerVolume;
			CoordType  InitialVoxelPerAtlas;
		};

		class IVDBBuilder
		{
		public:
			struct CreateParameters
			{
			};
			static std::unique_ptr<IVDBBuilder> Create(const CreateParameters& Params);
			virtual ~IVDBBuilder() {}

			struct FullBuildParameters
			{
				CoordType	  VoxelPerVolume;
				uint32_t	  EmptyScalarRangeNum;
				uint8_t*	  RAWVolumeData;
				glm::vec2*	  EmptyScalarRanges;
				VDBParameters VDBParams;
			};
			virtual void FullBuild(const FullBuildParameters& Params) = 0;
		};

	} // namespace VolData
} // namespace DepthBoxVDB

#endif
