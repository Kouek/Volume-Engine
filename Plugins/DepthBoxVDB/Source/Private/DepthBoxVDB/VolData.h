#ifndef PRIVATE_DEPTHBOXVDB_VOLDATA_H
#define PRIVATE_DEPTHBOXVDB_VOLDATA_H

#include <DepthBoxVDB/VolData.h>

#include <thread>
#include <memory>

#include <array>
#include <list>
#include <unordered_map>
#include <tuple>

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
				uint64_t X : 20;
				uint64_t Y : 20;
				uint64_t Z : 20;
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
			struct DataPerFrame
			{
				bool					  bDepthBoxUpdated = false;
				size_t					  PoolGPUMemInByte;
				std::vector<uint8_t>	  BrickedData;
				std::vector<BrickSortKey> BrickSortKeys;
			};

			struct ResidentDataPerFrame
			{
				enum class EEvent
				{
					TransferBrickDataToAtlas = 0,
					UpdateDepthBox,
					TransferBrickDataToCPU,
					BuildVDB,
					SwitchFrame,
					Num
				};
				std::array<cudaEvent_t, static_cast<uint32_t>(EEvent::Num)> Events;
				uint32_t													FrameIndex;
				std::list<uint32_t>::iterator								ResidentIndicesItr;

				VDBData* dVDBData = nullptr;
				std::array<thrust::device_vector<VDBNode>, VDBParameters::kMaxLevelNum>
					dNodePerLevels;
				std::array<thrust::device_vector<uint32_t>, VDBParameters::kMaxLevelNum - 1>
					dChildPerLevels;

				std::unordered_map<uint32_t, uint32_t> BrickWithFrameToAtlasBrick;

				ResidentDataPerFrame()
				{
					CUDA_CHECK(cudaMalloc(&dVDBData, sizeof(VDBData)));

					for (int32_t EventIndex = 0; EventIndex < static_cast<int32_t>(EEvent::Num);
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

					for (int32_t EventIndex = 0; EventIndex < static_cast<int32_t>(EEvent::Num);
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
					dVDBData = Other.dVDBData;
					Other.dVDBData = nullptr;
					dNodePerLevels = std::move(Other.dNodePerLevels);
					dChildPerLevels = std::move(Other.dChildPerLevels);

					BrickWithFrameToAtlasBrick = std::move(Other.BrickWithFrameToAtlasBrick);

					return *this;
				}
				void Invalidate(std::list<uint32_t>::iterator ResidentIndicesEnd)
				{
					FrameIndex = kInvalidIndex;
					ResidentIndicesItr = ResidentIndicesEnd;

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
				cudaError_t Wait(EEvent Event)
				{
					return CUDA_CHECK(cudaEventSynchronize(GetEvent(Event)));
				}
			};

			VDB(const CreateParameters& Params);
			~VDB();

			void FullBuild(const FullBuildParameters& Params) override;

			void StartAppendFrame(const StartAppendFrameParameters& Params) override;
			void AppendFrame(const AppendFrameParameters& Params) override;
			void EndAppendFrame();
			void RecacheResidentFrames(const RecacheResidentFramesParameters& Params) override;

			Status GetStatus() const override { return Status; }

			uint32_t GetFrameIndex() const override;
			uint32_t GetFrameNum() const override { return DataPerFrames.size(); }
			uint32_t GetMaxResidentFrameNum() const override { return MaxResidentFrameNum; }
			void	 SwitchToFrame(uint32_t FrameIndex) override;
			bool	 IsSwitched() const override;

			void UpdateDepthBox(const UpdateDepthBoxParameters& Params) override;

			const VDBParameters& GetVDBParameters() const { return VDBParams; }
			const VDBData*		 GetDeviceVDBData() const { return dVDBDataCurrentFrame; }
			cudaStream_t		 GetRenderStream() const { return getStream(EStream::Render); }

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
			void		switchFrame(ResidentDataPerFrame& RsdDataPF);
			static void switchFrameCUDAHostFunc(void* VDBPtr);

			void switchToFrame(uint32_t FrameIndex);
			void invalidateResidentFrames();
			void invalidate();
			void waitForAllTasks();

		private:
			enum class EStream
			{
				Copy = 0,
				Atlas,
				VDB,
				Render,
				Num
			};
			static constexpr std::array<unsigned int, static_cast<uint32_t>(EStream::Num)>
				kStreamFlags = { cudaStreamNonBlocking, cudaStreamNonBlocking, cudaStreamDefault,
					cudaStreamDefault };
			cudaStream_t getStream(EStream Stream) const
			{
				return Streams[static_cast<uint32_t>(Stream)];
			}

		private:
			size_t	  MaxAllowedGPUMemoryInByte = 0;
			uint32_t  MaxAllowedResidentFrameNum = 0;
			uint32_t  MaxResidentFrameNum;
			CoordType BrickPerAtlas;

			VDBData*							dVDBDataCurrentFrame;
			thrust::device_vector<BrickSortKey> dBrickSortKeys;
			std::list<uint32_t>::iterator		PlayingResidentFrameIndexItr;

			bool							bCanSwitchToFrameWorkerRun = true;
			uint32_t						FrameIndexToPlay;
			std::vector<uint32_t>			SwitchToFrameTasks;
			std::unique_ptr<std::thread>	SwitchToFrameWorker;
			mutable std::mutex				SwitchToFrameTasksMtx;
			mutable std::condition_variable SwitchToFrameTasksCV;

			VDBParameters VDBParams;

			std::array<cudaStream_t, static_cast<uint32_t>(EStream::Num)> Streams;

			std::shared_ptr<CUDA::Array>   AtlasArray;
			std::unique_ptr<CUDA::Texture> AtlasTexture;
			std::unique_ptr<CUDA::Surface> AtlasSurface;

			std::vector<glm::vec2>			 EmptyScalarRanges;
			std::vector<glm::vec2>			 EmptyScalarRangesReactive;
			thrust::device_vector<glm::vec2> dEmptyScalarRanges;

			std::vector<DataPerFrame> DataPerFrames;

			std::vector<ResidentDataPerFrame> ResidentDataPerFrames;
			std::vector<uint32_t>			  AvailableResidentIndices;
			std::list<uint32_t>				  ResidentIndices;

			std::vector<uint32_t>			AvailableAtlasBrick;
			std::vector<uint32_t>			AtlasBrickToBrickWithFrame;
			std::vector<uint32_t>			BrickWithFrameToAtlasBrick;
			thrust::device_vector<uint32_t> dAtlasBrickToBrickWithFrame;
			thrust::device_vector<uint32_t> dBrickWithFrameToAtlasBrick;

			Status Status;
		};

	} // namespace VolData
} // namespace DepthBoxVDB

#endif
