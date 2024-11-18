#ifndef PRIVATE_DEPTHBOXVDB_UTIL_H
#define PRIVATE_DEPTHBOXVDB_UTIL_H

#include <glm/glm.hpp>

#include <CUDA/Util.h>

namespace DepthBoxVDB
{
	struct Ray
	{
		glm::vec3 Origin;
		glm::vec3 Direction;
		float	  SceneDepthToPixel;

		struct HitShellResult
		{
			float tEnter;
			float tExit;
		};

		__host__ __device__ HitShellResult HitAABB(
			const glm::vec3& MinPosition, const glm::vec3& MaxPosition) const
		{
			HitShellResult Ret;

			float Tmp[6];
			Tmp[0] = (MinPosition.x - Origin.x) / Direction.x;
			Tmp[1] = (MaxPosition.x - Origin.x) / Direction.x;
			Tmp[2] = (MinPosition.y - Origin.y) / Direction.y;
			Tmp[3] = (MaxPosition.y - Origin.y) / Direction.y;
			Tmp[4] = (MinPosition.z - Origin.z) / Direction.z;
			Tmp[5] = (MaxPosition.z - Origin.z) / Direction.z;
			Ret.tEnter = glm::max(glm::max(glm::min(Tmp[0], Tmp[1]), glm::min(Tmp[2], Tmp[3])),
				glm::min(Tmp[4], Tmp[5]));
			Ret.tExit = glm::min(glm::min(glm::max(Tmp[0], Tmp[1]), glm::max(Tmp[2], Tmp[3])),
				glm::max(Tmp[4], Tmp[5]));
			Ret.tEnter = (Ret.tEnter < 0.f) ? 0.f : Ret.tEnter;

			return Ret;
		}
	};

	struct CUDA_ALIGN HDDA3D
	{
		glm::vec<3, int32_t> Sign;		 // Signs of Ray Direction
		glm::vec<3, int32_t> Mask;		 // 0 for should NOT and 1 for should move on XYZ axis
		glm::vec<3, int32_t> ChildCoord; // Position of child relative to its parent node
		float				 tCurr, tNext;
		glm::vec3			 tSide;		// Time that ray intersects with next plane in XYZ direction
		glm::vec3			 tDlt;		// Time delta
		const glm::vec3&	 Origin;	// Ray Origin
		const glm::vec3&	 Direction; // Ray Direction

		__host__ __device__ static HDDA3D Create(float tCurr, const Ray& InRay)
		{
			HDDA3D Ret = { .Origin = InRay.Origin, .Direction = InRay.Direction };
			Ret.Sign.x = Ret.Direction.x > 0.f ? 1 : Ret.Direction.x < 0.f ? -1 : 0;
			Ret.Sign.y = Ret.Direction.y > 0.f ? 1 : Ret.Direction.y < 0.f ? -1 : 0;
			Ret.Sign.z = Ret.Direction.z > 0.f ? 1 : Ret.Direction.z < 0.f ? -1 : 0;
			Ret.tCurr = tCurr;

			return Ret;
		}

		__host__ __device__ void Prepare(const glm::vec3& MinVoxelPosition, float ChildCoverVoxel)
		{
			tDlt = glm::abs(ChildCoverVoxel / Direction);
			glm::vec3 pFlt = (Origin + tCurr * Direction - MinVoxelPosition) / ChildCoverVoxel;
			tSide = ((glm::floor(pFlt) - pFlt + .5f) * glm::vec3{ Sign } + .5f) * tDlt + tCurr;
			ChildCoord = glm::floor(pFlt);
		}

		__host__ __device__ void Next()
		{
			Mask.x = static_cast<int32_t>((tSide.x < tSide.y) & (tSide.x <= tSide.z));
			Mask.y = static_cast<int32_t>((tSide.y < tSide.z) & (tSide.y <= tSide.x));
			Mask.z = static_cast<int32_t>((tSide.z < tSide.x) & (tSide.z <= tSide.y));
			tNext = Mask.x ? tSide.x : Mask.y ? tSide.y : Mask.z ? tSide.z : INFINITY;
		}

		__host__ __device__ void Step()
		{
			tCurr = tNext;
			tSide.x = isinf(tDlt.x) ? INFINITY : Mask.x ? tSide.x + tDlt.x : tSide.x;
			tSide.y = isinf(tDlt.y) ? INFINITY : Mask.y ? tSide.y + tDlt.y : tSide.y;
			tSide.z = isinf(tDlt.z) ? INFINITY : Mask.z ? tSide.z + tDlt.z : tSide.z;
			ChildCoord += Mask * Sign;
		}
	};

	struct CUDA_ALIGN DepthDDA2D
	{
		glm::vec<3, int32_t> Sign;
		glm::vec<3, int32_t> Mask;
		glm::vec<3, int32_t> CoordInBrick;
		float				 tCurr, tStart;
		float				 Depth;
		float				 tFromStart2Depth;
		glm::vec3			 tSide;
		glm::vec3			 tDelta;

		__host__ __device__ bool Init(float t, float VoxelPerBrick, int32_t MinDepCoordValInBrick,
			int32_t MaxDepCoordValInBrick, const glm::vec3& PosInBrick, const Ray& InRay)
		{
			Depth = 0.f;
			Sign.x = InRay.Direction.x > 0.f ? 1 : InRay.Direction.x < 0.f ? -1 : 0;
			Sign.y = InRay.Direction.y > 0.f ? 1 : InRay.Direction.y < 0.f ? -1 : 0;
			Sign.z = InRay.Direction.z > 0.f ? 1 : InRay.Direction.z < 0.f ? -1 : 0;
			tCurr = tStart = t;

			glm::ivec3 DepthSign;
			{
				glm::vec3 DistOnAxis(Sign.x == 0 ? INFINITY
						: Sign.x > 0			 ? PosInBrick.x
												 : VoxelPerBrick - PosInBrick.x,
					Sign.y == 0		 ? INFINITY
						: Sign.y > 0 ? PosInBrick.y
									 : VoxelPerBrick - PosInBrick.y,
					Sign.z == 0		 ? INFINITY
						: Sign.z > 0 ? PosInBrick.z
									 : VoxelPerBrick - PosInBrick.z);
				DepthSign.x =
					DistOnAxis.x < DistOnAxis.y && DistOnAxis.x <= DistOnAxis.z ? Sign.x : 0;
				DepthSign.y =
					DistOnAxis.y < DistOnAxis.z && DistOnAxis.y <= DistOnAxis.x ? Sign.y : 0;
				DepthSign.z =
					DistOnAxis.z < DistOnAxis.x && DistOnAxis.z <= DistOnAxis.y ? Sign.z : 0;

				if (DepthSign.x != 0 && DistOnAxis.x >= .5f)
					return false;
				if (DepthSign.y != 0 && DistOnAxis.y >= .5f)
					return false;
				if (DepthSign.z != 0 && DistOnAxis.z >= .5f)
					return false;
			}

			tDelta = glm::abs(1.f / InRay.Direction);
			CoordInBrick = glm::floor(PosInBrick);
#ifdef __CUDA_ARCH__
	#pragma unroll
#endif
			for (uint8_t Axis = 0; Axis < 3; ++Axis)
				if (CoordInBrick[Axis] < 0 || CoordInBrick[Axis] >= VoxelPerBrick)
					return false;

			tSide =
				((glm::floor(PosInBrick) - PosInBrick + .5f) * glm::vec3(Sign) + .5f) * tDelta + t;

			if (DepthSign.x != 0)
			{
				CoordInBrick.x = DepthSign.x == 1 ? MinDepCoordValInBrick : MaxDepCoordValInBrick;
				Sign.x = 0;
				tSide.x = INFINITY;
				tFromStart2Depth = glm::abs(InRay.Direction.x);
			}
			if (DepthSign.y != 0)
			{
				CoordInBrick.y = DepthSign.y == 1 ? MinDepCoordValInBrick : MaxDepCoordValInBrick;
				Sign.y = 0;
				tSide.y = INFINITY;
				tFromStart2Depth = glm::abs(InRay.Direction.y);
			}
			if (DepthSign.z != 0)
			{
				CoordInBrick.z = DepthSign.z == 1 ? MinDepCoordValInBrick : MaxDepCoordValInBrick;
				Sign.z = 0;
				tSide.z = INFINITY;
				tFromStart2Depth = glm::abs(InRay.Direction.z);
			}

			return true;
		}

		__host__ __device__ void StepNext()
		{
			Mask.x = static_cast<int32_t>((tSide.x < tSide.y) & (tSide.x <= tSide.z));
			Mask.y = static_cast<int32_t>((tSide.y < tSide.z) & (tSide.y <= tSide.x));
			Mask.z = static_cast<int32_t>((tSide.z < tSide.x) & (tSide.z <= tSide.y));

			tCurr = Mask.x ? tSide.x : Mask.y ? tSide.y : Mask.z ? tSide.z : INFINITY;
			Depth = tFromStart2Depth * (tCurr - tStart);

			tSide.x = isinf(tDelta.x) ? INFINITY : Mask.x ? tSide.x + tDelta.x : tSide.x;
			tSide.y = isinf(tDelta.y) ? INFINITY : Mask.y ? tSide.y + tDelta.y : tSide.y;
			tSide.z = isinf(tDelta.z) ? INFINITY : Mask.z ? tSide.z + tDelta.z : tSide.z;

			CoordInBrick += Mask * Sign;
		}
	};

} // namespace DepthBoxVDB

#endif
