#ifndef PUBLIC_DEPTHBOXVDB_VOLRENDERER_H
#define PUBLIC_DEPTHBOXVDB_VOLRENDERER_H

#include <memory>

#include <DepthBoxVDB/VolData.h>
#include <DepthBoxVDB/Util.h>

namespace DepthBoxVDB
{
	namespace VolRenderer
	{

		enum class ERHIType
		{
			D3D12 = 0,
			MAX
		};

		enum EVDBRenderTarget
		{
			Scene = 0,
			AABB0,
			AABB1,
			AABB2,
			DepthBox,
			PixelDepth,
			MAX
		};

		struct RendererParameters
		{
			int32_t	  MaxStepNum;
			bool	  bUsePreIntegratedTF;
			bool	  bUseDepthOcclusion;
			float	  Step;
			float	  MaxStepDist;
			float	  MaxAlpha;
			glm::vec3 InvVoxelSpaces;
		};

		struct RAWRendererParameters : RendererParameters
		{
			bool bUsePreIntegratedTF;
			bool bUseDepthOcclusion;
		};

		class IRenderer : Noncopyable
		{
		public:
			struct CreateParameters
			{
				ERHIType RHIType;
			};

			struct RegisterParameters
			{
				void* Device;
				void* InSceneDepthTexture;
				void* InOutColorTexture;
			};
			virtual void Register(const RegisterParameters& Params) = 0;
			virtual void Unregister() = 0;

			struct TransferFunctionParameters
			{
				const float* TransferFunctionData;
				const float* TransferFunctionDataPreIntegrated;
				uint32_t	 Resolution;
			};
			virtual void SetTransferFunction(const TransferFunctionParameters& Params) = 0;
		};

		class IVDBRenderer : virtual public IRenderer
		{
		public:
			struct CreateParameters : IRenderer::CreateParameters
			{
			};
			static std::unique_ptr<IVDBRenderer> Create(const CreateParameters& Params);
			virtual ~IVDBRenderer() {}

			struct RendererParameters : VolRenderer::RendererParameters
			{
				EVDBRenderTarget RenderTarget;
				bool			 bUseDepthBox;
				glm::vec3		 VisibleAABBMinPosition;
				glm::vec3		 VisibleAABBMaxPosition;
			};
			virtual void SetParameters(const RendererParameters& Params) = 0;

			struct RenderParameters
			{
				glm::mat4			 InverseProjection;
				glm::mat3			 CameraRotationToLocal;
				glm::vec3			 CameraPositionToVDB;
				const VolData::IVDB& VDB;
			};
			virtual void Render(const RenderParameters& Params) = 0;
		};

	} // namespace VolRenderer
} // namespace DepthBoxVDB

#endif
