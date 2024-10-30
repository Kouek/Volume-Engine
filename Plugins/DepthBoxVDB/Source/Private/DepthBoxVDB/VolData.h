#ifndef PRIVATE_DEPTHBOXVDB_VOLDATA_H
#define PRIVATE_DEPTHBOXVDB_VOLDATA_H

#include <DepthBoxVDB/VolData.h>

#include <memory>

#include <CUDA/Types.h>

#include "Util.h"

namespace DepthBoxVDB
{
	namespace VolData
	{
		struct CUDA_ALIGN VDBNode
		{
			CoordType CoordInVolume;
			CoordType CoordInAtlas;
			uint64_t  ChildOffset;
		};

		struct CUDA_ALIGN VDBData
		{
			VDBNode*  Nodes;
			uint32_t* Childs;

			cudaSurfaceObject_t AtlasSurface;
			cudaTextureObject_t AtlasTexture;

			VDBParameters VDBParams;
		};

		class VDBBuilder : public IVDBBuilder
		{
		public:
			VDBBuilder(const CreateParameters& Params);
			~VDBBuilder() {}

			void FullBuild(const FullBuildParameters& Params) override;

		private:
			void resizeAtlasArray(const FullBuildParameters& Params);

		private:
			cudaStream_t Stream = 0;

			std::shared_ptr<CUDA::Array>   AtlasArray;
			std::shared_ptr<CUDA::Texture> AtlasTexture;
			std::shared_ptr<CUDA::Surface> AtlasSuraface;
		};

	} // namespace VolData
} // namespace DepthBoxVDB

#endif
