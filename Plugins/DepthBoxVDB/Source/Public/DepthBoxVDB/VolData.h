#ifndef PUBLIC_DEPTHBOXVDB_VOLDATA_H
#define PUBLIC_DEPTHBOXVDB_VOLDATA_H

#include <numeric>
#include <memory>
#include <type_traits>

#include <array>

#include <glm/glm.hpp>

#include <DepthBoxVDB/Util.h>

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

		struct DEPTHBOXVDB_ALIGN VDBParameters
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
			static std::unique_ptr<IVDBBuilder> Create();
			virtual ~IVDBBuilder() {}

			struct FullBuildParameters
			{
				uint8_t*	  RAWVolumeData;
				VDBParameters VDBParams;
			};
			virtual void FullBuild(const FullBuildParameters& Params) = 0;
		};

	} // namespace VolData
} // namespace DepthBoxVDB

#endif
