#ifndef PRIVATE_DEPTHBOXVDB_VOLRENDERER_H
#define PRIVATE_DEPTHBOXVDB_VOLRENDERER_H

#include <DepthBoxVDB/VolRenderer.h>

#include <CUDA/Types.h>

#include "VolData.h"
#include "D3D12Util.h"
#include "Util.h"

namespace DepthBoxVDB
{
	namespace VolRenderer
	{
		constexpr float Eps = .01f;

		class VDBRenderer : public IVDBRenderer
		{
		public:
			VDBRenderer(const CreateParameters& Params);
			~VDBRenderer();

			void Register(const RegisterParameters& Params) override;
			void Unregister() override;

			void SetParameters(const VDBRendererParameters& Params) override;
			void SetTransferFunction(const TransferFunctionParameters& Params) override;

			void Render(const RenderParameters& Params) override;
			template <typename VoxelType, bool bUseDepthBox, bool bUsePreIntegratedTF>
			void render(const RenderParameters& Params, const VolData::VDBData* dVDBData);

		private:
			ERHIType RHIType;

			UINT				   D3D12NodeMask;
			cudaStream_t		   Stream = 0;
			VDBRendererParameters* dParams = nullptr;

			bool	   bUseDepthBox = false;
			bool	   bUsePreIntegratedTF = false;
			glm::uvec2 RenderResolution;

			std::unique_ptr<D3D12::TextureMappedCUDASurface> InSceneDepthTexture;
			std::unique_ptr<D3D12::TextureMappedCUDASurface> OutColorTexture;
			std::unique_ptr<CUDA::Texture>					 TransferFunctionTexture;
			std::unique_ptr<CUDA::Texture>					 TransferFunctionTexturePreIntegrated;
		};

	} // namespace VolRenderer
} // namespace DepthBoxVDB

#endif
