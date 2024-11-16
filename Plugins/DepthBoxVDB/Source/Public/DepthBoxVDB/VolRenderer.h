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
			None = 0,
			D3D12,
			MAX
		};

		enum ERenderTarget
		{
			Scene = 0,
			AABB0,
			AABB1,
			AABB2,
			DepthBox,
			MAX
		};

		struct CUDA_ALIGN VDBRendererParameters
		{
			ERenderTarget RenderTarget;
			int32_t		  MaxStepNum;
			bool		  bUseDepthBox;
			bool		  bUsePreIntegratedTF;
			float		  Step;
			float		  MaxStepDist;
			float		  MaxAlpha;
		};

		class IVDBRenderer : Noncopyable
		{
		public:
			struct CreateParameters
			{
				ERHIType RHIType;
			};
			static std::unique_ptr<IVDBRenderer> Create(const CreateParameters& Params);
			virtual ~IVDBRenderer() {}

			struct RegisterParameters
			{
				void* Device;
				void* InDepthTexture;
				void* OutColorTexture;
			};
			virtual void Register(const RegisterParameters& Params) = 0;
			virtual void Unregister() = 0;

			virtual void SetParameters(const VDBRendererParameters& Params) = 0;

			struct TransferFunctionParameters
			{
				const float* TransferFunctionData;
				const float* TransferFunctionDataPreIntegrated;
				uint32_t	 Resolution;
			};
			virtual void SetTransferFunction(const TransferFunctionParameters& Params) = 0;

			struct RenderParameters
			{
				glm::mat4					InverseProjection;
				glm::mat3					CameraRotationToLocal;
				glm::vec3					CameraPositionToLocal;
				const VolData::IVDBBuilder& Builder;
			};
			virtual void Render(const RenderParameters& Params) = 0;
		};

	} // namespace VolRenderer
} // namespace DepthBoxVDB

#endif
