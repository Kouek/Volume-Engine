#include "VolRenderer.h"

std::unique_ptr<DepthBoxVDB::VolRenderer::IVDBRenderer>
DepthBoxVDB::VolRenderer::IVDBRenderer::Create(const CreateParameters& Params)
{
	if (Params.RHIType == ERHIType::D3D12)
	{
		return std::make_unique<VDBRenderer>(Params);
	}
	return {};
}

struct CUDA_ALIGN VDBStack
{
	DepthBoxVDB::VolData::VDBNode nodes[DepthBoxVDB::VolData::VDBParameters::MaxLevelNum - 1];
	const DepthBoxVDB::VolData::VDBData& VDBData;
	float	tExits[DepthBoxVDB::VolData::VDBParameters::MaxLevelNum - 1];
	int32_t Level;

	__host__ __device__ static VDBStack Create(const DepthBoxVDB::VolData::VDBData& VDBData)
	{
		VDBStack Stack = { .VDBData = VDBData, .Level = VDBData.VDBParams.RootLevel + 1 };
		return Stack;
	}
	__host__ __device__ void Push(uint32_t NodeIndex, float tExit)
	{
		--Level;
		nodes[Level - 1] = VDBData.Node(Level, NodeIndex);
		tExits[Level - 1] = tExit;
	}
	__host__ __device__ DepthBoxVDB::VolData::VDBNode& TopNode() { return nodes[Level - 1]; }
	__host__ __device__ float						   TopTExit() { return tExits[Level - 1]; }
	__host__ __device__ void						   Pop() { ++Level; }
	__host__ __device__ bool Empty() { return Level == VDBData.VDBParams.RootLevel + 1; }
	__host__ __device__ bool Full() { return Level == 0; }
};

DepthBoxVDB::VolRenderer::VDBRenderer::VDBRenderer(const CreateParameters& Params)
	: RHIType(Params.RHIType)
{
	int DeviceNum = 0;
	CUDA_CHECK(cudaGetDeviceCount(&DeviceNum));
	assert(DeviceNum > 0);

	cudaDeviceProp Prop;
	CUDA_CHECK(cudaGetDeviceProperties(&Prop, 0));
	D3D12NodeMask = Prop.luidDeviceNodeMask;

	CUDA_CHECK(cudaStreamCreate(&Stream));
}

DepthBoxVDB::VolRenderer::VDBRenderer::~VDBRenderer()
{
	if (dParams)
	{
		CUDA_CHECK(cudaFree(dParams));
	}
}

void DepthBoxVDB::VolRenderer::VDBRenderer::Register(const RegisterParameters& Params)
{
	ID3D12Device*	Device = reinterpret_cast<ID3D12Device*>(Params.Device);
	ID3D12Resource* InDepthTextureNative = reinterpret_cast<ID3D12Resource*>(Params.InDepthTexture);
	ID3D12Resource* OutColorTextureNative =
		reinterpret_cast<ID3D12Resource*>(Params.OutColorTexture);
	InDepthTexture = std::make_unique<D3D12::TextureMappedCUDASurface>(
		D3D12NodeMask, Device, InDepthTextureNative);
	OutColorTexture = std::make_unique<D3D12::TextureMappedCUDASurface>(
		D3D12NodeMask, Device, OutColorTextureNative);
	RenderResolution.x = InDepthTexture->TextureDesc.Width;
	RenderResolution.y = InDepthTexture->TextureDesc.Height;
	assert(RenderResolution.x == OutColorTexture->TextureDesc.Width
		&& RenderResolution.y == OutColorTexture->TextureDesc.Height);
}

void DepthBoxVDB::VolRenderer::VDBRenderer::Unregister()
{
	InDepthTexture.reset();
	OutColorTexture.reset();
}

void DepthBoxVDB::VolRenderer::VDBRenderer::SetParameters(const VDBRendererParameters& Params)
{
	bUseDepthBox = Params.bUseDepthBox;
	bUsePreIntegratedTF = Params.bUsePreIntegratedTF;

	if (!dParams)
	{
		CUDA_CHECK(cudaMalloc(&dParams, sizeof(VDBRendererParameters)));
	}
	CUDA_CHECK(cudaMemcpy(dParams, &Params, sizeof(VDBRendererParameters), cudaMemcpyHostToDevice));
}

void DepthBoxVDB::VolRenderer::VDBRenderer::SetTransferFunction(
	const TransferFunctionParameters& Params)
{
	auto Create = [&](const float* Data, const glm::uvec3& Dim) {
		auto Arr = std::make_shared<CUDA::Array>(reinterpret_cast<const float4*>(Data), Dim);

		cudaTextureDesc TexDesc{};
		TexDesc.normalizedCoords = 1;
		TexDesc.filterMode = cudaFilterModeLinear;
		TexDesc.addressMode[0] = TexDesc.addressMode[1] = TexDesc.addressMode[2] =
			cudaAddressModeBorder;
		TexDesc.readMode = cudaReadModeElementType;
		return std::make_unique<CUDA::Texture>(Arr, TexDesc);
	};
	TransferFunctionTexture =
		Create(Params.TransferFunctionData, glm::uvec3(Params.Resolution, 1, 1));
	TransferFunctionTexturePreIntegrated = Create(Params.TransferFunctionDataPreIntegrated,
		glm::uvec3(Params.Resolution, Params.Resolution, 1));
}

__device__ static DepthBoxVDB::Ray GenRay(const glm::uvec3& DispatchThreadID,
	const glm::uvec2& RenderResolution, const glm::mat4& InverseProjection,
	const glm::mat3& CameraRotation, const glm::vec3& CameraPosition)
{
	DepthBoxVDB::Ray EyeRay;

	// Map [0, RenderResolution.xy - 1] to (-1, 1)
	glm::vec4 Tmp;
	Tmp.z = RenderResolution.x;
	Tmp.w = RenderResolution.y;
	Tmp.x = (2.f * DispatchThreadID.x + 1.f - Tmp.z) / Tmp.z;
	Tmp.y = (2.f * (RenderResolution.y - 1 - DispatchThreadID.y) + 1.f - Tmp.w) / Tmp.w;

	// Inverseproject
	Tmp.z = 1.f;
	Tmp.w = 1.f;
	Tmp = InverseProjection * Tmp;

	EyeRay.Direction = Tmp;
	EyeRay.Direction = CameraRotation * glm::normalize(EyeRay.Direction);

	EyeRay.Origin = CameraPosition;

	return EyeRay;
}

struct CUDA_ALIGN LeafEnteredParameters
{
	float						  tEnter;
	float						  tExit;
	DepthBoxVDB::VolData::VDBNode Node;
};

template <typename OnChildPushedType, typename OnSteppedType, typename LeafEnteredType>
struct RayCastVDBCallbacks
{
	OnChildPushedType OnChildPushed;
	OnSteppedType	  OnStepped;
	LeafEnteredType	  LeafEntered;
};

template <typename OnChildPushedType, typename OnSteppedType, typename LeafEnteredType>
__device__ static glm::vec4 RayCastVDB(const DepthBoxVDB::VolData::VDBData& VDBData,
	const DepthBoxVDB::Ray&													EyeRay,
	RayCastVDBCallbacks<OnChildPushedType, OnSteppedType, LeafEnteredType>	Callbacks)
{
	using namespace DepthBoxVDB;

	const VolData::VDBParameters& VDBParams = VDBData.VDBParams;

	Ray::HitShellResult HitShell = EyeRay.HitAABB(glm::vec3(0.f), VDBParams.VoxelPerVolume);

	if (HitShell.tEnter >= HitShell.tExit)
	{
		return glm::vec4(0.f);
	}

	VDBStack Stack = VDBStack::Create(VDBData);
	Stack.Push(0, HitShell.tExit - VolRenderer::Eps);

	HDDA3D Hdda3d = HDDA3D ::Create(HitShell.tEnter + VolRenderer::Eps, EyeRay);
	Hdda3d.Prepare(glm::vec3(0.f), VDBParams.ChildCoverVoxelPerLevels[VDBParams.RootLevel]);

	while (!Stack.Empty() && [&]() {
#ifdef __CUDA_ARCH__
	#pragma unroll
#endif
		for (uint8_t Axis = 0; Axis < 3; ++Axis)
			if (Hdda3d.ChildCoord[Axis] < 0
				|| Hdda3d.ChildCoord[Axis] >= VDBParams.ChildPerLevels[Stack.Level])
				return false;
		return true;
	}())
	{
		Hdda3d.Next();

		auto& Curr = Stack.TopNode();
		auto  ChildIndexInCurr = VDBData.Child(Stack.Level, Hdda3d.ChildCoord, Curr);

		if (ChildIndexInCurr != VolData::VDBData::InvalidChild)
		{
			if (Stack.Level == 1)
			{
				Hdda3d.tCurr += VolRenderer::Eps;

				if constexpr (!std::is_same_v<LeafEnteredType, nullptr_t>)
				{
					LeafEnteredParameters LeafEntered;
					LeafEntered.tEnter = Hdda3d.tCurr;
					LeafEntered.tExit = Hdda3d.tNext - VolRenderer::Eps;
					LeafEntered.Node = VDBData.Node(0, ChildIndexInCurr);
					if (Callbacks.LeafEntered(LeafEntered))
						break;
				}

				Hdda3d.Step();
			}
			else
			{
				Stack.Push(ChildIndexInCurr, Hdda3d.tNext - VolRenderer::Eps);
				Hdda3d.tCurr += VolRenderer::Eps;
				Hdda3d.Prepare(
					Stack.TopNode().Coord, VDBParams.ChildCoverVoxelPerLevels[Stack.Level]);

				if constexpr (!std::is_same_v<OnChildPushedType, nullptr_t>)
				{
					Callbacks.OnChildPushed();
				}
			}
		}
		else
		{
			Hdda3d.Step();

			if constexpr (!std::is_same_v<OnSteppedType, nullptr_t>)
			{
				Callbacks.OnStepped();
			}
		}

		while (Hdda3d.tCurr >= Stack.TopTExit())
		{
			Stack.Pop();
			if (Stack.Empty())
				break;

			Hdda3d.Prepare(Stack.TopNode().Coord, VDBParams.ChildCoverVoxelPerLevels[Stack.Level]);
		}
	}
}

template <typename T>
__device__ bool DepthSkip(const glm::vec3& PosInBrick,
	const DepthBoxVDB::VolData::CoordType& MinCoordInAtlasBrick, LeafEnteredParameters& Params,
	const DepthBoxVDB::VolData::VDBData& VDBData, const DepthBoxVDB::Ray& EyeRay)
{
	using namespace DepthBoxVDB;

	const VolData::VDBParameters& VDBParams = VDBData.VDBParams;

	DepthDDA2D DepDda2d;
	if (!DepDda2d.Init(Params.tEnter, VDBParams.ChildPerLevels[0],
			VDBParams.DepthCoordValueInAtlasBrick[0], VDBParams.DepthCoordValueInAtlasBrick[1],
			PosInBrick, EyeRay))
		return false;

	while (true)
	{
		T Depth = surf3Dread<T>(VDBData.AtlasSurface,
			sizeof(T) * (MinCoordInAtlasBrick.x + DepDda2d.CoordInBrick.x),
			MinCoordInAtlasBrick.y + DepDda2d.CoordInBrick.x,
			MinCoordInAtlasBrick.z + DepDda2d.CoordInBrick.z);
		if (Depth <= DepDda2d.Depth)
			break;
		if (DepDda2d.tCurr >= Params.tExit)
			return true;

		Params.tEnter = DepDda2d.tCurr;
		DepDda2d.StepNext();
	}
	return false;
}

template <typename VoxelType, bool bUseDepthBox, bool bUsePreIntegratedTF>
__device__ static glm::vec4 RenderScene(cudaTextureObject_t TransferFunctionTexture,
	const DepthBoxVDB::VolRenderer::VDBRendererParameters&	RendererParams,
	const DepthBoxVDB::VolData::VDBData& VDBData, const DepthBoxVDB::Ray& EyeRay)
{
	using namespace DepthBoxVDB;

	const VolData::VDBParameters& VDBParams = VDBData.VDBParams;

	glm::vec3 Color(0.f);
	float	  Alpha = 0.f;
	float	  ScalarPrev = -1.f;

	glm::vec3 DeltaPos = RendererParams.Step * EyeRay.Direction;
	int32_t	  StepNum = 1;

	RayCastVDBCallbacks Callbacks = { /* OnChildPushed */ nullptr,
		/* OnStepped */
		[&]() { ScalarPrev = -1.f; },
		/* LeafEntered */
		[&](LeafEnteredParameters& Params) {
			Params.tEnter = RendererParams.Step * glm::ceil(Params.tEnter / RendererParams.Step);
			glm::vec3	MinPosInBrick = glm::vec3(Params.Node.Coord * VDBParams.ChildPerLevels[0]);
			glm::vec3	PosInBrick = EyeRay.Origin + Params.tEnter * EyeRay.Direction - MinPosInBrick;
			VolData::CoordType MinCoordInAtlasBrick =
				Params.Node.CoordInAtlas * VDBParams.VoxelPerAtlasBrick
				+ VDBParams.ApronAndDepthWidth;

			if constexpr (bUseDepthBox)
			{
				if (DepthSkip<VoxelType>(PosInBrick, MinCoordInAtlasBrick, Params, VDBData, EyeRay))
					return true;

				Params.tEnter =
					RendererParams.Step * glm::ceil(Params.tEnter / RendererParams.Step);
				PosInBrick = EyeRay.Origin + Params.tEnter * EyeRay.Direction - MinPosInBrick;
			}

			glm::vec3	MinPosInAtlasBrick(MinCoordInAtlasBrick);
			while (Params.tEnter < Params.tExit && Params.tEnter <= RendererParams.MaxStepDist
				&& StepNum <= RendererParams.MaxStepNum && [&]() {
#ifdef __CUDA_ARCH__
	#pragma unroll
#endif
					   for (uint8_t Axis = 0; Axis < 3; ++Axis)
						   if (PosInBrick[Axis] < 0.f
							   || PosInBrick[Axis] >= VDBParams.ChildPerLevels[0])
							   return false;
					   return true;
				   }())
			{
				glm::vec3 SamplePos = MinPosInAtlasBrick + PosInBrick;
				float	  Scalar =
					tex3D<float>(VDBData.AtlasTexture, SamplePos.x, SamplePos.y, SamplePos.z);
				if (ScalarPrev < 0.f)
					ScalarPrev = Scalar;

				if constexpr (bUsePreIntegratedTF)
				{
					float4 TFColorAlpha =
						tex2D<float4>(TransferFunctionTexture, ScalarPrev, Scalar);
					Color = Color
						+ (1.f - Alpha) * glm::vec3(TFColorAlpha.x, TFColorAlpha.y, TFColorAlpha.z);
					Alpha = Alpha + (1.f - Alpha) * TFColorAlpha.w;

					ScalarPrev = Scalar;
				}
				else
				{
					float4 TFColorAlpha = tex2D<float4>(TransferFunctionTexture, Scalar, 0.f);
					Color = Color
						+ (1.f - Alpha) * TFColorAlpha.w
							* glm::vec3(TFColorAlpha.x, TFColorAlpha.y, TFColorAlpha.z);
					Alpha = Alpha + (1.f - Alpha) * TFColorAlpha.w;
				}

				if (Alpha >= RendererParams.MaxAlpha)
					return true;

				Params.tEnter += RendererParams.Step;
				PosInBrick += DeltaPos;
				++StepNum;
			}

			return false;
		} };
	RayCastVDB(VDBData, EyeRay, Callbacks);

	return glm::vec4(Color, Alpha);
}

__device__ static glm::vec4 RenderAABB(
	const DepthBoxVDB::VolData::VDBData& VDBData, const DepthBoxVDB::Ray& EyeRay)
{
	using namespace DepthBoxVDB;

	const VolData::VDBParameters& VDBParams = VDBData.VDBParams;

	glm::vec3 Color(0.f);
	float	  Alpha = 0.f;

	RayCastVDBCallbacks Callbacks = { /* OnChildPushed */ nullptr,
		/* OnStepped */ nullptr,
		/* LeafEntered */
		[&](const LeafEnteredParameters& Params) {
			glm::vec3 PosInBrick = EyeRay.Origin + Params.tEnter * EyeRay.Direction
				- glm::vec3(Params.Node.Coord * VDBParams.ChildPerLevels[0]);

			Color = PosInBrick / float(VDBParams.ChildPerLevels[0]);
			Alpha = 1.f;

			return true; // Break at the first leaf entered
		} };
	RayCastVDB(VDBData, EyeRay, Callbacks);

	return glm::vec4(Color, Alpha);
}

void DepthBoxVDB::VolRenderer::VDBRenderer::Render(const RenderParameters& Params)
{
	if (!InDepthTexture || !OutColorTexture)
	{
		std::cerr << "Empty Mapped CUDA Texture/Surface(s)\n";
		return;
	}
	if (!InDepthTexture->IsComplete() || !OutColorTexture->IsComplete())
	{
		std::cerr << "Incomplete Mapped CUDA Texture/Surface(s)\n";
		return;
	}
	if (!TransferFunctionTexture || !TransferFunctionTexturePreIntegrated)
	{
		std::cerr << "Empty CUDA Texture/Surface(s)\n";
		return;
	}
	if (!TransferFunctionTexture->IsComplete()
		|| !TransferFunctionTexturePreIntegrated->IsComplete())
	{
		std::cerr << "Incomplete CUDA Texture/Surface(s)\n";
		return;
	}
	if (!dParams)
	{
		std::cerr << "Empty dParams\n";
		return;
	}

	const VolData::VDBBuilder&	Builder = static_cast<const VolData::VDBBuilder&>(Params.Builder);
	const VolData::VDBProvider* Provider = Builder.Provider.get();
	const VolData::VDBData*		dVDBData = Builder.GetDeviceData();
	if (!Provider || !dVDBData)
	{
		std::cerr << "Empty Provider or Device Data\n";
		return;
	}

	const VolData::VDBParameters& VDBParams = Provider->VDBParams;

	switch (VDBParams.VoxelType)
	{
		case VolData::EVoxelType::UInt8:
			if (bUseDepthBox && bUsePreIntegratedTF)
				render<uint8_t, true, true>(Params, dVDBData);
			else if (!bUseDepthBox && bUsePreIntegratedTF)
				render<uint8_t, false, true>(Params, dVDBData);
			else if (bUseDepthBox && !bUsePreIntegratedTF)
				render<uint8_t, true, false>(Params, dVDBData);
			else
				render<uint8_t, false, false>(Params, dVDBData);
			break;
		case VolData::EVoxelType::Float32:
			if (bUseDepthBox && bUsePreIntegratedTF)
				render<float, true, true>(Params, dVDBData);
			else if (!bUseDepthBox && bUsePreIntegratedTF)
				render<float, false, true>(Params, dVDBData);
			else if (bUseDepthBox && !bUsePreIntegratedTF)
				render<float, true, false>(Params, dVDBData);
			else
				render<float, false, false>(Params, dVDBData);
			break;
	}

	CUDA_CHECK(cudaStreamSynchronize(Stream));
}

template <typename VoxelType, bool bUseDepthBox, bool bUsePreIntegratedTF>
void DepthBoxVDB::VolRenderer::VDBRenderer::render(
	const RenderParameters& Params, const VolData::VDBData* dVDBData)
{
	auto RenderKernel = [InverseProjection = Params.InverseProjection,
							CameraRotation = Params.CameraRotationToLocal,
							CameraPosition = Params.CameraPositionToLocal,
							RenderResolution = RenderResolution,
							InDepthSurface = InDepthTexture->SurfaceObject,
							OutColorSurface = OutColorTexture->SurfaceObject,
							TransferFunctionTexture = bUsePreIntegratedTF
								? TransferFunctionTexturePreIntegrated->Get()
								: TransferFunctionTexture->Get(),
							VDBData = dVDBData,
							RendererParams =
								dParams] __device__(const glm::uvec3& DispatchThreadID) {
		if (DispatchThreadID.x >= RenderResolution.x || DispatchThreadID.y >= RenderResolution.y)
			return;

		Ray EyeRay = GenRay(
			DispatchThreadID, RenderResolution, InverseProjection, CameraRotation, CameraPosition);

		glm::vec4 Color;
		switch (RendererParams->RenderTarget)
		{
			case ERenderTarget::Scene:
				Color = RenderScene<VoxelType, bUseDepthBox, bUsePreIntegratedTF>(
					TransferFunctionTexture, *RendererParams, *VDBData, EyeRay);
				break;
			case ERenderTarget::AABB0:
				Color = RenderAABB(*VDBData, EyeRay);
				break;
		}

		Color = glm::clamp(Color * 255.f, 0.f, 255.f);
		uchar4 ColorUCh4{ Color.r, Color.g, Color.b, Color.a };

		surf2Dwrite(
			ColorUCh4, OutColorSurface, sizeof(uchar4) * DispatchThreadID.x, DispatchThreadID.y);
	};

	dim3 ThreadPerBlock(CUDA::ThreadPerBlockX2D, CUDA::ThreadPerBlockY2D, 1);
	dim3 BlockPerGrid((RenderResolution.x + ThreadPerBlock.x - 1) / ThreadPerBlock.x,
		(RenderResolution.y + ThreadPerBlock.y - 1) / ThreadPerBlock.y);
	CUDA::ParallelFor(BlockPerGrid, ThreadPerBlock, RenderKernel, Stream);
}
template void DepthBoxVDB::VolRenderer::VDBRenderer::render<uint8_t, true, true>(
	const RenderParameters& Params, const VolData::VDBData* dVDBData);
template void DepthBoxVDB::VolRenderer::VDBRenderer::render<uint8_t, false, true>(
	const RenderParameters& Params, const VolData::VDBData* dVDBData);
template void DepthBoxVDB::VolRenderer::VDBRenderer::render<uint8_t, true, false>(
	const RenderParameters& Params, const VolData::VDBData* dVDBData);
template void DepthBoxVDB::VolRenderer::VDBRenderer::render<uint8_t, false, false>(
	const RenderParameters& Params, const VolData::VDBData* dVDBData);
template void DepthBoxVDB::VolRenderer::VDBRenderer::render<float, true, true>(
	const RenderParameters& Params, const VolData::VDBData* dVDBData);
template void DepthBoxVDB::VolRenderer::VDBRenderer::render<float, false, true>(
	const RenderParameters& Params, const VolData::VDBData* dVDBData);
template void DepthBoxVDB::VolRenderer::VDBRenderer::render<float, true, false>(
	const RenderParameters& Params, const VolData::VDBData* dVDBData);
template void DepthBoxVDB::VolRenderer::VDBRenderer::render<float, false, false>(
	const RenderParameters& Params, const VolData::VDBData* dVDBData);
