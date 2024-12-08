#ifndef PRIVATE_DEPTHBOXVDB_VOLDATA_H
#define PRIVATE_DEPTHBOXVDB_VOLDATA_H

#include <DepthBoxVDB/VolData.h>

#include <memory>

#include <array>
#include <queue>
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
			__host__ __device__ bool operator<(const BrickSortKey& Other) const
			{
				return Key < Other.Key;
			}
		};

		class VDB : public IVDB
		{
		public:
			struct PopupFrameParameters
			{
				DepthBoxVDB::VolData::VDBData** Src = nullptr;
				DepthBoxVDB::VolData::VDBData** Dst = nullptr;
			};

			VDB(const CreateParameters& Params);
			~VDB();

			void FullBuild(const FullBuildParameters& Params) override;

			void StartAppendFrame(const StartAppendFrameParameters& Params) override;
			void AppendFrame(const AppendFrameParameters& Params) override;
			void EndAppendFrame();

			uint32_t GetFrameNum() { return DataPerFrames.size(); }
			void	 SwitchToFrame(uint32_t FrameIndex) override;

			void UpdateDepthBox(const UpdateDepthBoxParameters& Params) override;

			const VDBParameters& GetVDBParameters() const { return VDBParams; }
			VDBData*			 GetDeviceVDBData() const { return dVDBDataCurrentFrame; }

			uint32_t BrickCoordToIndex(const CoordWithFrameType& CoordWithFrame)
			{
				return CoordWithFrameToIndex(CoordWithFrame, VDBParams.BrickPerVolume);
			}
			CoordWithFrameType BrickIndexToCoord(uint32_t BrickIndexWithFrame)
			{
				return IndexToCoordWithFrame(BrickIndexWithFrame, VDBParams.BrickPerVolume);
			}
			uint32_t AtlasBrickCoordToIndex(const CoordWithFrameType& CoordWithFrame)
			{
				return CoordWithFrameToIndex(CoordWithFrame, BrickPerAtlas);
			}
			CoordWithFrameType AtlasBrickIndexToCoord(uint32_t BrickIndex)
			{
				return IndexToCoordWithFrame(BrickIndex, BrickPerAtlas);
			}

			// Declare Private functions in Public scope to use CUDA Lambda
			void generateDataPerFrame(const uint8_t* RAWVolumeData, uint32_t FrameIndex);
			bool allocateResource();

			void							   transferBrickDataToAtlas(uint32_t ResidentIndex);
			void							   updateDepthBox(uint32_t ResidentIndex);
			template <typename VoxelType> void updateDepthBox(uint32_t FrameIndex);
			void							   transferBrickDataToCPU(uint32_t ResidentIndex);
			void							   buildVDB(uint32_t ResidentIndex);

		private:
			enum class EStream
			{
				Copy = 0,
				Atlas,
				VDB,
				Host,
				Max
			};
			cudaStream_t getStream(EStream Stream)
			{
				return Streams[static_cast<uint32_t>(Stream)];
			}

		private:
			struct DataPerFrame
			{
				bool								bUpdatedFromEmptyScalarRanges = false;
				bool								bTransferredToCPU = false;
				std::vector<uint8_t>				BrickedData;
				std::vector<BrickSortKey>			BrickSortKeys;
				thrust::device_vector<BrickSortKey> dBrickSortKeys;
			};
			struct ResidentDataPerFrame
			{
				enum class EEvent
				{
					TransferBrickDataToAtlas = 0,
					UpdateDepthBox,
					TransferBrickDataToCPU,
					BuildVDB,
					Max
				};
				std::array<cudaEvent_t, static_cast<uint32_t>(EEvent::Max)> Events;
				uint32_t													FrameIndex;

				VDBData* dVDBData = nullptr;
				std::array<thrust::device_vector<VDBNode>, VDBParameters::kMaxLevelNum>
					dNodePerLevels;
				std::array<thrust::device_vector<uint32_t>, VDBParameters::kMaxLevelNum>
					dChildPerLevels;

				std::unordered_map<uint32_t, uint32_t> BrickWithFrameToAtlasBrick;

				ResidentDataPerFrame()
				{
					CUDA_CHECK(cudaMalloc(&dVDBData, sizeof(VDBData)));

					for (int32_t EventIndex = 0; EventIndex < static_cast<int32_t>(EEvent::Max);
						 ++EventIndex)
					{
						CUDA_CHECK(cudaEventCreate(&Events[EventIndex]));
					}
				}
				~ResidentDataPerFrame()
				{
					if (dVDBData)
					{
						CUDA_CHECK(cudaFree(dVDBData));
					}

					for (int32_t EventIndex = 0; EventIndex < static_cast<int32_t>(EEvent::Max);
						 ++EventIndex)
					{
						if (Events[EventIndex] == 0)
							continue;
						CUDA_CHECK(cudaEventDestroy(Events[EventIndex]));
					}
				}
				ResidentDataPerFrame(const ResidentDataPerFrame&) = delete;
				ResidentDataPerFrame& operator=(const ResidentDataPerFrame&) = delete;
				ResidentDataPerFrame(ResidentDataPerFrame&& Other) { operator=(std::move(Other)); }
				ResidentDataPerFrame& operator=(ResidentDataPerFrame&& Other)
				{
					Events = Other.Events;
					FrameIndex = Other.FrameIndex;
					dNodePerLevels = std::move(Other.dNodePerLevels);
					dChildPerLevels = std::move(Other.dChildPerLevels);

					return *this;
				}

				void Invalidate()
				{
					FrameIndex = kInvalidIndex;

					BrickWithFrameToAtlasBrick.clear();

					for (auto& dNodePerLevel : dNodePerLevels)
						dNodePerLevel.clear();
					for (auto& dChildPerLevel : dChildPerLevels)
						dChildPerLevel.clear();
				}

				cudaEvent_t GetEvent(EEvent Event) { return Events[static_cast<uint32_t>(Event)]; }
				cudaError_t Record(EEvent Event, cudaStream_t Stream)
				{
					return CUDA_CHECK(cudaEventRecord(GetEvent(Event), Stream));
				}
				cudaError_t Wait(cudaStream_t Stream, EEvent Event)
				{
					return CUDA_CHECK(cudaStreamWaitEvent(Stream, GetEvent(Event)));
				}
			};

			size_t	  MaxAllowedGPUMemoryInByte = 0;
			uint32_t  MaxAllowedResidentFrameNum = 0;
			uint32_t  MaxResidentFrameNum = 0;
			uint32_t  ResidentFrameNum = 0;
			CoordType BrickPerAtlas;

			std::array<cudaStream_t, static_cast<uint32_t>(EStream::Max)> Streams;

			VDBData*			 dVDBDataCurrentFrame = nullptr;
			VDBParameters		 VDBParams;
			PopupFrameParameters PopupFrameParams;

			std::shared_ptr<CUDA::Array>   AtlasArray;
			std::unique_ptr<CUDA::Texture> AtlasTexture;
			std::unique_ptr<CUDA::Surface> AtlasSurface;

			std::vector<glm::vec2>			 EmptyScalarRanges;
			std::vector<glm::vec2>			 EmptyScalarRangesReactive;
			thrust::device_vector<glm::vec2> dEmptyScalarRanges;

			std::vector<DataPerFrame> DataPerFrames;

			/*
			 * Restriction:
			 * 1. ResidentDataPerFrames.size() == MaxResidentFrameNum
			 * 2. Real size of ResidentDataPerFrames is ResidentFrameNum
			 * 3. ResidentDataPerFrames[ResidentIndices[0...MaxResidentFrameNum - 1]].FrameIndex
			 *    == 1st, 2nd, ..., MaxResidentFrameNum-th playing frames
			 */
			std::vector<ResidentDataPerFrame> ResidentDataPerFrames;
			std::deque<uint32_t>			  ResidentIndices;

			std::vector<uint32_t>			AvailableAtlasBrick;
			std::vector<uint32_t>			AtlasBrickToBrickWithFrame;
			std::vector<uint32_t>			BrickWithFrameToAtlasBrick;
			thrust::device_vector<uint32_t> dAtlasBrickToBrickWithFrame;
			thrust::device_vector<uint32_t> dBrickWithFrameToAtlasBrick;
		};

	} // namespace VolData
} // namespace DepthBoxVDB

#endif
