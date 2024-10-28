#ifndef PRIVATE_DEPTHBOXVDB_VOLDATA_H
#define PRIVATE_DEPTHBOXVDB_VOLDATA_H

#include <DepthBoxVDB/VolData.h>

#include <cuda.h>

namespace DepthBoxVDB
{
	namespace VolData
	{
		struct DEPTHBOXVDB_ALIGN VDBNode
		{
			CoordType CoordInVolume;
			CoordType CoordInAtlas;
			uint64_t  ChildOffset;
		};

		struct DEPTHBOXVDB_ALIGN VDBData
		{
			VDBNode*  Nodes;
			uint32_t* Childs;

			cudaSurfaceObject_t AtlasSurface;
			cudaTextureObject_t AtlasTexture;

			VDBParameters VDBParams;
		};

		class VDBBuilderImpl : public IVDBBuilder
		{
		public:
			~VDBBuilderImpl() {}

			void FullBuild(const FullBuildParameters& Params) override;
		};

	} // namespace VolData
} // namespace DepthBoxVDB

#endif
