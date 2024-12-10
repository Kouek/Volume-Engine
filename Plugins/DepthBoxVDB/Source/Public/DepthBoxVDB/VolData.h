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
	using CoordValueType = int32_t;
	using CoordType = glm::vec<3, CoordValueType>;

	using CoordWithFrameType = glm::vec<4, CoordValueType>;

	static constexpr CoordValueType kInvalidCoordValue = std::numeric_limits<CoordValueType>::max();
	static constexpr uint32_t		kInvalidIndex = std::numeric_limits<uint32_t>::max();

	inline __device__ __host__ uint32_t CoordToIndex(
		const CoordType& Coord, const CoordType& Extent)
	{
		return static_cast<uint32_t>(Coord.z) * Extent.y * Extent.x + Coord.y * Extent.x + Coord.x;
	}
	inline __device__ __host__ uint32_t CoordWithFrameToIndex(
		const CoordWithFrameType& CoordWithFrame, const CoordType& Extent)
	{
		return static_cast<uint32_t>(CoordWithFrame.w) * Extent.z * Extent.y * Extent.x
			+ CoordWithFrame.z * Extent.y * Extent.x + CoordWithFrame.y * Extent.x
			+ CoordWithFrame.x;
	}
	inline __device__ __host__ CoordType IndexToCoord(uint32_t Index, const CoordType& Extent)
	{
		CoordType Coord;
		uint32_t  Tmp = Extent.y * Extent.x;
		Coord.z = Index / Tmp;
		Tmp = Index - Coord.z * Tmp;
		Coord.y = Tmp / Extent.x;
		Coord.x = Tmp - Coord.y * Extent.x;

		return Coord;
	}
	inline __device__ __host__ CoordWithFrameType IndexToCoordWithFrame(
		uint32_t Index, const CoordType& Extent)
	{
		CoordWithFrameType CoordWithFrame;
		uint32_t		   Tmp1 = Extent.y * Extent.x;
		uint32_t		   Tmp = Extent.z * Tmp1;
		CoordWithFrame.w = Index / Tmp;
		Tmp = Index - CoordWithFrame.w * Tmp;
		CoordWithFrame.z = Tmp / Tmp1;
		Tmp = Tmp - CoordWithFrame.z * Tmp1;
		CoordWithFrame.y = Tmp / Extent.x;
		CoordWithFrame.x = Tmp - CoordWithFrame.y * Extent.x;

		return CoordWithFrame;
	}

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

	namespace VolData
	{

		struct CUDA_ALIGN RAWVolumeParameters
		{
			EVoxelType VoxelType;
			CoordType  VoxelPerVolume;
		};

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

		class IVDB : Noncopyable
		{
		public:
			struct CreateParameters
			{
			};
			static std::shared_ptr<IVDB> Create(const CreateParameters& Params);
			virtual ~IVDB() {}

			struct FullBuildParameters
			{
				const uint8_t*	 RAWVolumeData;
				const glm::vec2* EmptyScalarRanges;
				uint32_t		 EmptyScalarRangeNum;
				uint32_t		 MaxAllowedGPUMemoryInGB;
				uint32_t		 MaxAllowedResidentFrameNum;
				VDBParameters	 VDBParams;
			};
			virtual void FullBuild(const FullBuildParameters& Params) = 0;

			struct StartAppendFrameParameters
			{
				const glm::vec2* EmptyScalarRanges;
				uint32_t		 EmptyScalarRangeNum;
				uint32_t		 MaxAllowedGPUMemoryInGB;
				uint32_t		 MaxAllowedResidentFrameNum;
				VDBParameters	 VDBParams;
			};
			virtual void StartAppendFrame(const StartAppendFrameParameters& Params) = 0;
			struct AppendFrameParameters
			{
				const uint8_t* RAWVolumeData;
			};
			virtual void AppendFrame(const AppendFrameParameters& Params) = 0;
			virtual void EndAppendFrame() = 0;

			virtual uint32_t GetFrameNum() = 0;
			virtual uint32_t GetMaxResidentFrameNum() = 0;
			virtual void	 SwitchToFrame(uint32_t FrameIndex) = 0;

			struct UpdateDepthBoxParameters
			{
				const glm::vec2* EmptyScalarRanges;
				uint32_t		 EmptyScalarRangeNum;
			};
			virtual void UpdateDepthBox(const UpdateDepthBoxParameters& Params) = 0;
		};

	} // namespace VolData
} // namespace DepthBoxVDB

#endif
