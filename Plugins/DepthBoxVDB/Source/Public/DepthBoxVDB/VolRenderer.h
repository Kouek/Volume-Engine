#ifndef PUBLIC_DEPTHBOXVDB_VOLRENDERER_H
#define PUBLIC_DEPTHBOXVDB_VOLRENDERER_H

#include <memory>

#include <DepthBoxVDB/VolData.h>

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

		class IVDBRenderer
		{
		public:
			struct CreateParameters
			{
				ERHIType RHIType;
			};
			static std::unique_ptr<IVDBRenderer> Create(const CreateParameters& Params);
			virtual ~IVDBRenderer() {}

			struct RendererParameters
			{
				void* Device;
				void* InDepthTexture;
				void* OutColorTexture;
			};
			virtual void Register(const RendererParameters& Params) = 0;
			virtual void Unregister() = 0;

			struct RenderParameters
			{
				bool bUseDepthBox;
			};
			virtual void Render(const RenderParameters& Params) = 0;
		};

	} // namespace VolRenderer
} // namespace DepthBoxVDB

#endif
