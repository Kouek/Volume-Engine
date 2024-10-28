#ifndef PRIVATE_DEPTHBOXVDB_VOLRENDERER_H
#define PRIVATE_DEPTHBOXVDB_VOLRENDERER_H

#include <DepthBoxVDB/VolRenderer.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "D3D12Util.h"

namespace DepthBoxVDB
{
	namespace VolRenderer
	{
		class VDBRendererImpl : public IVDBRenderer
		{
		public:
			VDBRendererImpl(ERHIType RHIType);
			~VDBRendererImpl() {}

			void Register(const RendererParameters& Params) override;
			void Unregister() override;
			void Render(const RenderParameters& Params) override;

		private:
			ERHIType RHIType;

			UINT		 D3D12NodeMask;
			cudaStream_t Stream;

			glm::uvec2 RenderResolution;

			std::unique_ptr<D3D12TextureInteropCUDA> InDepthTexture;
			std::unique_ptr<D3D12TextureInteropCUDA> OutColorTexture;
		};

	} // namespace VolRenderer
} // namespace DepthBoxVDB

#endif
