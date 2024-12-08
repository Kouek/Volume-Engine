#ifndef PRIVATE_DEPTHBOXVDB_VOLRENDERER_H
#define PRIVATE_DEPTHBOXVDB_VOLRENDERER_H

#include <DepthBoxVDB/VolRenderer.h>

#include <unordered_map>

#include <CUDA/Types.h>

#include "VolData.h"
#include "D3D12Util.h"
#include "Util.h"

namespace DepthBoxVDB
{
	namespace D3D12
	{

		class RendererSharedStates
		{
		public:
			static RendererSharedStates& Instance();

			void Register(void* Device, void* ExternalTexture);
			void Unregister(void* RegisteredTexture);

		private:
			RendererSharedStates();
			~RendererSharedStates();

		public:
			UINT		 D3D12NodeMask;
			cudaStream_t Stream = 0;

			std::unordered_map<void*, std::shared_ptr<TextureMappedCUDASurface>>
				ExternalToRegisteredTextures;
		};

	} // namespace D3D12
} // namespace DepthBoxVDB

namespace DepthBoxVDB
{
	namespace VolRenderer
	{
		constexpr float Eps = .01f;

		class Renderer : virtual public IRenderer
		{
		public:
			Renderer(const CreateParameters& Params);
			virtual ~Renderer();

			void Register(const RegisterParameters& Params) override;
			void Unregister() override;

			void SetTransferFunction(const TransferFunctionParameters& Params) override;

		protected:
			cudaStream_t Stream = 0;
			ERHIType	 RHIType;

			/* Keep only necessary states in CPU */
			bool	   bUseDepthBox = false;
			bool	   bUsePreIntegratedTF = false;
			glm::uvec2 RenderResolution;

			std::shared_ptr<D3D12::TextureMappedCUDASurface> InSceneDepthTexture;
			std::shared_ptr<D3D12::TextureMappedCUDASurface> InOutColorTexture;
			std::unique_ptr<CUDA::Texture>					 TransferFunctionTexture;
			std::unique_ptr<CUDA::Texture>					 TransferFunctionTexturePreIntegrated;
		};

		class VDBRenderer : virtual public IVDBRenderer, public Renderer
		{
		public:
			struct CUDA_ALIGN DeviceRendererParameters
			{
				EVDBRenderTarget RenderTarget;
				int32_t			 MaxStepNum;
				bool			 bUseDepthOcclusion;
				float			 Step;
				float			 MaxStepDist;
				float			 MaxAlpha;
				glm::vec3		 InvVoxelSpaces;
				glm::vec3		 VisibleAABBMinPosition;
				glm::vec3		 VisibleAABBMaxPosition;
			};

			VDBRenderer(const CreateParameters& Params);
			~VDBRenderer();

			void SetParameters(const RendererParameters& Params) override;

			void Render(const RenderParameters& Params) override;
			template <typename VoxelType, bool bUseDepthBox, bool bUsePreIntegratedTF>
			void render(const RenderParameters& Params, const VolData::VDBData* dVDBData);

		private:
			DeviceRendererParameters* dParams = nullptr;
		};

	} // namespace VolRenderer
} // namespace DepthBoxVDB

#endif
