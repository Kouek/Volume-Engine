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
	DepthBoxVDB::VolData::VDBNode nodes[DepthBoxVDB::VolData::VDBParameters::kMaxLevelNum - 1];
	const DepthBoxVDB::VolData::VDBData& VDBData;
	float	tExits[DepthBoxVDB::VolData::VDBParameters::kMaxLevelNum - 1];
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
	: Renderer(Params)
{
}

DepthBoxVDB::VolRenderer::VDBRenderer::~VDBRenderer()
{
	if (dParams)
	{
		CUDA_CHECK(cudaFree(dParams));
	}
}

void DepthBoxVDB::VolRenderer::VDBRenderer::SetParameters(const RendererParameters& Params)
{
	bUseDepthBox = Params.bUseDepthBox;
	bUsePreIntegratedTF = Params.bUsePreIntegratedTF;

	if (!dParams)
	{
		CUDA_CHECK(cudaMalloc(&dParams, sizeof(DeviceRendererParameters)));
	}

	DeviceRendererParameters ParamsInternal;

#define ASSIGN(Member) ParamsInternal.Member = Params.Member;

	ASSIGN(RenderTarget)
	ASSIGN(MaxStepNum)
	ASSIGN(bUseDepthOcclusion)
	ASSIGN(Step)
	ASSIGN(MaxStepDist)
	ASSIGN(MaxAlpha)
	ASSIGN(InvVoxelSpaces)
	ASSIGN(VisibleAABBMinPosition)
	ASSIGN(VisibleAABBMaxPosition)

#undef ASSIGN

	CUDA_CHECK(cudaMemcpy(
		dParams, &ParamsInternal, sizeof(DeviceRendererParameters), cudaMemcpyHostToDevice));
}

__device__ static DepthBoxVDB::Ray GenRay(const glm::uvec3& DispatchThreadID,
	const glm::uvec2& RenderResolution, const glm::mat4& InverseProjection,
	const glm::mat3& CameraRotationToLocal, const glm::vec3& CameraPositionVDBSpace,
	const glm::vec3& InvVoxelSpaces)
{
	DepthBoxVDB::Ray EyeRay;

	// Map [0, RenderResolution.xy - 1] to (-1, 1)
	{
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
	}

	EyeRay.Direction = glm::normalize(EyeRay.Direction);
	EyeRay.SceneDepthToPixel = glm::abs(EyeRay.Direction.z);
	EyeRay.Direction = InvVoxelSpaces * (CameraRotationToLocal * EyeRay.Direction);
	EyeRay.SceneDepthToPixel = glm::length(EyeRay.Direction) / EyeRay.SceneDepthToPixel;
	EyeRay.Direction = glm::normalize(EyeRay.Direction);

	EyeRay.Origin = CameraPositionVDBSpace;

	return EyeRay;
}

struct CUDA_ALIGN IntersectionTestVDBParameters
{
	DepthBoxVDB::Ray::HitShellResult& HitShell;
};

struct CUDA_ALIGN OnChildPushedParameters
{
	float								 tEnter;
	float								 tExit;
	int32_t								 Level;
	const DepthBoxVDB::VolData::VDBNode& Node;
};

struct CUDA_ALIGN LeafEnteredParameters
{
	float								 tEnter;
	float								 tExit;
	const DepthBoxVDB::VolData::VDBNode& Node;
};

template <typename IntersectionTestType, typename OnChildPushedType, typename OnSteppedType,
	typename LeafEnteredType>
struct RayCastVDBCallbacks
{
	IntersectionTestType IntersectionTest;
	OnChildPushedType	 OnChildPushed;
	OnSteppedType		 OnStepped;
	LeafEnteredType		 LeafEntered;
};

template <typename IntersectionTestType, typename OnChildPushedType, typename OnSteppedType,
	typename LeafEnteredType>
__device__ static glm::vec4 RayCastVDB(const DepthBoxVDB::VolData::VDBData& VDBData,
	const DepthBoxVDB::Ray&													EyeRay,
	RayCastVDBCallbacks<IntersectionTestType, OnChildPushedType, OnSteppedType, LeafEnteredType>
		Callbacks)
{
	using namespace DepthBoxVDB;

	const VolData::VDBParameters& VDBParams = VDBData.VDBParams;

	Ray::HitShellResult HitShell;
	if constexpr (!std::is_same_v<IntersectionTestType, nullptr_t>)
	{
		IntersectionTestVDBParameters Params{ HitShell };
		Callbacks.IntersectionTest(Params);
	}
	if (HitShell.tEnter >= HitShell.tExit)
	{
		return glm::vec4(0.f);
	}

	VDBStack Stack = VDBStack::Create(VDBData);
	Stack.Push(0, HitShell.tExit - VolRenderer::Eps);
	HDDA3D Hdda3d = HDDA3D ::Create(HitShell.tEnter + VolRenderer::Eps, EyeRay);
	Hdda3d.Prepare(glm::vec3(0.f), VDBParams.ChildCoverVoxelPerLevels[VDBParams.RootLevel]);

	if constexpr (!std::is_same_v<OnChildPushedType, nullptr_t>)
	{
		OnChildPushedParameters Params{ .tEnter = Hdda3d.tCurr,
			.tExit = Hdda3d.tNext - VolRenderer::Eps,
			.Level = Stack.Level,
			.Node = Stack.TopNode() };
		Callbacks.OnChildPushed(Params);
	}

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

		auto& Parent = Stack.TopNode();
		auto  ChildIndex = VDBData.Child(Stack.Level, Hdda3d.ChildCoord, Parent);

		if (ChildIndex != VolData::VDBData::kInvalidChild)
		{
			if (Stack.Level == 1)
			{
				Hdda3d.tCurr += VolRenderer::Eps;

				if constexpr (!std::is_same_v<LeafEnteredType, nullptr_t>)
				{
					LeafEnteredParameters Params{ .tEnter = Hdda3d.tCurr,
						.tExit = Hdda3d.tNext - VolRenderer::Eps,
						.Node = VDBData.Node(0, ChildIndex) };
					if (Callbacks.LeafEntered(Params))
						break;
				}

				Hdda3d.Step();
				if constexpr (!std::is_same_v<OnSteppedType, nullptr_t>)
				{
					Callbacks.OnStepped();
				}
			}
			else
			{
				Stack.Push(ChildIndex, Hdda3d.tNext - VolRenderer::Eps);
				Hdda3d.tCurr += VolRenderer::Eps;
				Hdda3d.Prepare(
					Stack.TopNode().Coord * VDBParams.ChildCoverVoxelPerLevels[Stack.Level + 1],
					VDBParams.ChildCoverVoxelPerLevels[Stack.Level]);

				if constexpr (!std::is_same_v<OnChildPushedType, nullptr_t>)
				{
					OnChildPushedParameters Params{ .tEnter = Hdda3d.tCurr,
						.tExit = Hdda3d.tNext - VolRenderer::Eps,
						.Level = Stack.Level,
						.Node = Stack.TopNode() };
					Callbacks.OnChildPushed(Params);
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

			Hdda3d.Prepare(Stack.Level == VDBParams.RootLevel
					? VolData::CoordType(0)
					: Stack.TopNode().Coord * VDBParams.ChildCoverVoxelPerLevels[Stack.Level + 1],
				VDBParams.ChildCoverVoxelPerLevels[Stack.Level]);
		}
	}
}

template <typename VoxelType>
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
		VoxelType Depth = surf3Dread<VoxelType>(VDBData.AtlasSurface,
			sizeof(VoxelType) * (MinCoordInAtlasBrick.x + DepDda2d.CoordInBrick.x),
			MinCoordInAtlasBrick.y + DepDda2d.CoordInBrick.y,
			MinCoordInAtlasBrick.z + DepDda2d.CoordInBrick.z);
		if (Depth <= DepDda2d.Depth + VolRenderer::Eps)
			break;
		if (DepDda2d.tCurr >= Params.tExit)
			return true;

		Params.tEnter = DepDda2d.tCurr;
		DepDda2d.StepNext();
	}
	return false;
}

template <typename VoxelType, bool bUseDepthBox, bool bUsePreIntegratedTF>
__device__ static glm::vec4 RenderScene(cudaTextureObject_t				   TransferFunctionTexture,
	float																   InputPixelDepth,
	const DepthBoxVDB::VolRenderer::VDBRenderer::DeviceRendererParameters& RendererParams,
	const DepthBoxVDB::VolData::VDBData& VDBData, const DepthBoxVDB::Ray& EyeRay)
{
	using namespace DepthBoxVDB;

	const VolData::VDBParameters& VDBParams = VDBData.VDBParams;

	glm::vec3 Color(0.f);
	float	  Alpha = 0.f;
	float	  ScalarPrev = -1.f;

	glm::vec3 DeltaPos = RendererParams.Step * EyeRay.Direction;
	int32_t	  StepNum = 1;

	glm::vec3 AABBMinPosition = glm::max(glm::vec3(0.f), RendererParams.VisibleAABBMinPosition);
	glm::vec3 AABBMaxPosition =
		glm::min(glm::vec3(VDBParams.VoxelPerVolume), RendererParams.VisibleAABBMaxPosition);

	RayCastVDBCallbacks Callbacks = { /* IntersectionTest */
		[&](IntersectionTestVDBParameters& Params) {
			Params.HitShell = EyeRay.HitAABB(AABBMinPosition, AABBMaxPosition);
		},
		/* OnChildPushed */ nullptr,
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
					return false;

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
				if (Params.tEnter >= InputPixelDepth)
					return true;

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
		}
	};
	RayCastVDB(VDBData, EyeRay, Callbacks);

	return glm::vec4(Color, Alpha);
}

__device__ static glm::vec4 RenderAABB(
	int32_t Level, const DepthBoxVDB::VolData::VDBData& VDBData, const DepthBoxVDB::Ray& EyeRay)
{
	using namespace DepthBoxVDB;

	const VolData::VDBParameters& VDBParams = VDBData.VDBParams;

	glm::vec3 Color(0.f);
	float	  Alpha = 0.f;

	RayCastVDBCallbacks Callbacks = { /* IntersectionTest */
		[&](IntersectionTestVDBParameters& Params) {
			Params.HitShell = EyeRay.HitAABB(glm::vec3(0.f),
				glm::vec3(VDBParams.ChildPerLevels[VDBParams.RootLevel]
					* VDBParams.ChildCoverVoxelPerLevels[VDBParams.RootLevel]));
		},
		/* OnChildPushed */
		[&](const OnChildPushedParameters& Params) {
			if (Level != Params.Level)
				return;

			int32_t		VoxelPerNode = Level == VDBParams.RootLevel
					? VDBParams.ChildCoverVoxelPerLevels[Level] * VDBParams.ChildPerLevels[Level]
					: VDBParams.ChildCoverVoxelPerLevels[Level + 1];
			glm::vec3	PosInBrick = EyeRay.Origin + Params.tEnter * EyeRay.Direction
				- glm::vec3(Params.Node.Coord * VoxelPerNode);

			Color = Color + (1.f - Alpha) * .5f * PosInBrick / float(VoxelPerNode);
			Alpha = Alpha + (1.f - Alpha) * .5f;
		},
		/* OnStepped */ nullptr,
		/* LeafEntered */
		[&](const LeafEnteredParameters& Params) {
			if (Level != 0)
				return false;

			glm::vec3	PosInBrick = EyeRay.Origin + Params.tEnter * EyeRay.Direction
				- glm::vec3(Params.Node.Coord * VDBParams.ChildPerLevels[0]);

			Color = Color + (1.f - Alpha) * .5f * PosInBrick / float(VDBParams.ChildPerLevels[0]);
			Alpha = Alpha + (1.f - Alpha) * .5f;

			return false;
		}
	};
	RayCastVDB(VDBData, EyeRay, Callbacks);

	return glm::vec4(Color, Alpha);
}

template <typename VoxelType>
__device__ static glm::vec4 RenderDepthBox(
	const DepthBoxVDB::VolRenderer::VDBRenderer::DeviceRendererParameters& RendererParams,
	const DepthBoxVDB::VolData::VDBData& VDBData, const DepthBoxVDB::Ray& EyeRay)
{
	using namespace DepthBoxVDB;

	const VolData::VDBParameters& VDBParams = VDBData.VDBParams;

	glm::vec3 Color(0.f);
	float	  Alpha = 0.f;

	RayCastVDBCallbacks Callbacks = { /* IntersectionTest */
		[&](IntersectionTestVDBParameters& Params) {
			Params.HitShell = EyeRay.HitAABB(glm::vec3(0.f),
				glm::vec3(VDBParams.ChildPerLevels[VDBParams.RootLevel]
					* VDBParams.ChildCoverVoxelPerLevels[VDBParams.RootLevel]));
		},
		/* OnChildPushed */ nullptr,
		/* OnStepped */ nullptr,
		/* LeafEntered */
		[&](LeafEnteredParameters& Params) {
			Params.tEnter = RendererParams.Step * glm::ceil(Params.tEnter / RendererParams.Step);
			glm::vec3	MinPosInBrick = glm::vec3(Params.Node.Coord * VDBParams.ChildPerLevels[0]);
			glm::vec3	PosInBrick = EyeRay.Origin + Params.tEnter * EyeRay.Direction - MinPosInBrick;
			VolData::CoordType MinCoordInAtlasBrick =
				Params.Node.CoordInAtlas * VDBParams.VoxelPerAtlasBrick
				+ VDBParams.ApronAndDepthWidth;

			Alpha = 1.f;
			DepthDDA2D	DepDda2d;
			if (DepDda2d.Init(Params.tEnter, VDBParams.ChildPerLevels[0],
					VDBParams.DepthCoordValueInAtlasBrick[0],
					VDBParams.DepthCoordValueInAtlasBrick[1], PosInBrick, EyeRay))
			{
				float Depth = surf3Dread<VoxelType>(VDBData.AtlasSurface,
					sizeof(VoxelType) * (MinCoordInAtlasBrick.x + DepDda2d.CoordInBrick.x),
					MinCoordInAtlasBrick.y + DepDda2d.CoordInBrick.y,
					MinCoordInAtlasBrick.z + DepDda2d.CoordInBrick.z);
				Color = glm::vec3(Depth / float(VDBParams.ChildPerLevels[0]));

				// Debug FaceIndex
				// if (Depth == 0 || Depth == 1)
				//	Color.r = Color.g = 1.f;
				// else if (Depth == 2 || Depth == 3)
				//	Color.g = 1.f;
				// else if (Depth == 4 || Depth == 5)
				//	Color.b = 1.f;
			}
			else
			{
				Color.r = 1.f;
			}

			return true; // Break at the first leaf entered
		}
	};
	RayCastVDB(VDBData, EyeRay, Callbacks);

	return glm::vec4(Color, Alpha);
}

__device__ static glm::vec4 RenderPixelDepth(float InputPixelDepth)
{
	using namespace DepthBoxVDB;

	int32_t Division = glm::ceil(InputPixelDepth / 255.f);
	Division = glm::min(Division, 3);

	float	  Remained = InputPixelDepth;
	glm::vec3 Color(0.f);
	for (int32_t i = 0; i < Division; ++i)
	{
		Color[i] = glm::min(Remained, 255.f);
		Remained -= Color[i];
	}
	Color /= 255.f;

	return glm::vec4(Color, 1.f);
}

void DepthBoxVDB::VolRenderer::VDBRenderer::Render(const RenderParameters& Params)
{
	if (!InSceneDepthTexture || !InOutColorTexture)
	{
		std::cerr << "Empty Mapped CUDA Texture/Surface(s)\n";
		return;
	}
	if (!InSceneDepthTexture->IsComplete() || !InOutColorTexture->IsComplete())
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

	const VolData::VDB&		VDB = static_cast<const VolData::VDB&>(Params.VDB);
	const VolData::VDBData* dVDBData = VDB.GetDeviceData();
	if (!dVDBData)
	{
		std::cerr << "Empty Device Data\n";
		return;
	}

	const VolData::VDBParameters& VDBParams = VDB.VDBParams;

	auto DispatchRender = [&]<typename VoxelType>(VoxelType*) {
		if (bUseDepthBox && bUsePreIntegratedTF)
			render<VoxelType, true, true>(Params, dVDBData);
		else if (!bUseDepthBox && bUsePreIntegratedTF)
			render<VoxelType, false, true>(Params, dVDBData);
		else if (bUseDepthBox && !bUsePreIntegratedTF)
			render<VoxelType, true, false>(Params, dVDBData);
		else
			render<VoxelType, false, false>(Params, dVDBData);
	};

	switch (VDBParams.VoxelType)
	{
		case VolData::EVoxelType::UInt8:
			DispatchRender((uint8_t*)nullptr);
			break;
		case VolData::EVoxelType::UInt16:
			DispatchRender((uint16_t*)nullptr);
			break;
		case VolData::EVoxelType::Float32:
			DispatchRender((float*)nullptr);
			break;
	}

	CUDA_CHECK(cudaStreamSynchronize(Stream));
}

template <typename VoxelType, bool bUseDepthBox, bool bUsePreIntegratedTF>
void DepthBoxVDB::VolRenderer::VDBRenderer::render(
	const RenderParameters& Params, const VolData::VDBData* dVDBData)
{
	auto RenderKernel = [InverseProjection = Params.InverseProjection,
							CameraRotationToLocal = Params.CameraRotationToLocal,
							CameraPositionToVDB = Params.CameraPositionToVDB,
							RenderResolution = RenderResolution,
							InSceneDepthSurface = InSceneDepthTexture
								? InSceneDepthTexture->SurfaceObject
								: cudaSurfaceObject_t(0),
							OutColorSurface = InOutColorTexture->SurfaceObject,
							TransferFunctionTexture = bUsePreIntegratedTF
								? TransferFunctionTexturePreIntegrated->Get()
								: TransferFunctionTexture->Get(),
							VDBData = dVDBData,
							RendererParams =
								dParams] __device__(const glm::uvec3& DispatchThreadID) {
		if (DispatchThreadID.x >= RenderResolution.x || DispatchThreadID.y >= RenderResolution.y)
			return;

		Ray EyeRay = GenRay(DispatchThreadID, RenderResolution, InverseProjection,
			CameraRotationToLocal, CameraPositionToVDB, RendererParams->InvVoxelSpaces);

		auto GetPixelDepth = [&]() {
			if (InSceneDepthSurface == 0 || !RendererParams->bUseDepthOcclusion)
				return 3.4028234e38f; // std::numeric_limits<float>::max()

			return EyeRay.SceneDepthToPixel
				* surf2Dread<float>(
					InSceneDepthSurface, sizeof(float) * DispatchThreadID.x, DispatchThreadID.y);
		};

		glm::vec4 Color;
		switch (RendererParams->RenderTarget)
		{
			case EVDBRenderTarget::Scene:
			{
				float InputPixelDepth = GetPixelDepth();
				Color = RenderScene<VoxelType, bUseDepthBox, bUsePreIntegratedTF>(
					TransferFunctionTexture, InputPixelDepth, *RendererParams, *VDBData, EyeRay);
			}
			break;
			case EVDBRenderTarget::AABB0:
			case EVDBRenderTarget::AABB1:
			case EVDBRenderTarget::AABB2:
				Color = RenderAABB(static_cast<int32_t>(RendererParams->RenderTarget)
						- static_cast<int32_t>(EVDBRenderTarget::AABB0),
					*VDBData, EyeRay);
				break;
			case EVDBRenderTarget::DepthBox:
				Color = RenderDepthBox<VoxelType>(*RendererParams, *VDBData, EyeRay);
				break;
			case EVDBRenderTarget::PixelDepth:
			{
				float InputPixelDepth = GetPixelDepth();
				// Debug tExit
				// Ray::HitShellResult HitShell =
				//	EyeRay.HitAABB(glm::vec3(0.f), VDBData->VDBParams.VoxelPerVolume);
				// if (HitShell.tEnter >= HitShell.tExit)
				//{
				//	Color = glm::vec4(0.f);
				//	break;
				//}
				// InputPixelDepth = HitShell.tExit;

				Color = RenderPixelDepth(InputPixelDepth);
			}
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
template void DepthBoxVDB::VolRenderer::VDBRenderer::render<uint16_t, true, true>(
	const RenderParameters& Params, const VolData::VDBData* dVDBData);
template void DepthBoxVDB::VolRenderer::VDBRenderer::render<uint16_t, false, true>(
	const RenderParameters& Params, const VolData::VDBData* dVDBData);
template void DepthBoxVDB::VolRenderer::VDBRenderer::render<uint16_t, true, false>(
	const RenderParameters& Params, const VolData::VDBData* dVDBData);
template void DepthBoxVDB::VolRenderer::VDBRenderer::render<uint16_t, false, false>(
	const RenderParameters& Params, const VolData::VDBData* dVDBData);
template void DepthBoxVDB::VolRenderer::VDBRenderer::render<float, true, true>(
	const RenderParameters& Params, const VolData::VDBData* dVDBData);
template void DepthBoxVDB::VolRenderer::VDBRenderer::render<float, false, true>(
	const RenderParameters& Params, const VolData::VDBData* dVDBData);
template void DepthBoxVDB::VolRenderer::VDBRenderer::render<float, true, false>(
	const RenderParameters& Params, const VolData::VDBData* dVDBData);
template void DepthBoxVDB::VolRenderer::VDBRenderer::render<float, false, false>(
	const RenderParameters& Params, const VolData::VDBData* dVDBData);
