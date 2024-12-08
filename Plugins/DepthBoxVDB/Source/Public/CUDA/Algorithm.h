#ifndef PUBLIC_CUDA_ALGORITHM_H
#define PUBLIC_CUDA_ALGORITHM_H

#include <cuda.h>

#include <vector>

#include <thrust/scan.h>
#include <thrust/system/cuda/execution_policy.h>

#include <thrust/device_vector.h>

namespace CUDA
{

	template <typename IdxTy, typename T>
	thrust::device_vector<IdxTy> Difference(
		const thrust::device_vector<T>& dSrcs, IdxTy SrcNum = 0, cudaStream_t Stream = 0)
	{
		thrust::device_vector<IdxTy> dDiffs(SrcNum == 0 ? dSrcs.size() : SrcNum, 1);
		thrust::for_each(thrust::cuda::par_nosync.on(Stream),
			thrust::make_counting_iterator(IdxTy(1)),
			thrust::make_counting_iterator(
				SrcNum == 0 ? static_cast<IdxTy>(dDiffs.size()) : SrcNum),
			[Diffs = thrust::raw_pointer_cast(dDiffs.data()),
				Srcs = thrust::raw_pointer_cast(dSrcs.data())] __device__(IdxTy SrcIdx) {
				Diffs[SrcIdx] = Srcs[SrcIdx - 1] == Srcs[SrcIdx] ? 0 : 1;
			});
		return dDiffs;
	}

	template <typename IdxTy>
	thrust::device_vector<IdxTy> CompactIndexes(
		IdxTy SrcNum, const thrust::device_vector<IdxTy>& dValids, cudaStream_t Stream = 0)
	{
		thrust::device_vector<IdxTy> dCmpctPrefixSums(SrcNum);
		thrust::inclusive_scan(thrust::cuda::par_nosync.on(Stream), dValids.begin(), dValids.end(),
			dCmpctPrefixSums.begin());
		auto CmpctNum = dCmpctPrefixSums.back();

		thrust::device_vector<IdxTy> dCmpctIdxs(CmpctNum);
		thrust::for_each(thrust::cuda::par_nosync.on(Stream),
			thrust::make_counting_iterator(IdxTy(1)), thrust::make_counting_iterator(SrcNum),
			[CmpctIdxs = thrust::raw_pointer_cast(dCmpctIdxs.data()),
				CmpctPrefixSums =
					thrust::raw_pointer_cast(dCmpctPrefixSums.data())] __device__(IdxTy SrcIdx) {
				auto PrefixSum = CmpctPrefixSums[SrcIdx];
				if (PrefixSum != CmpctPrefixSums[SrcIdx - 1])
					CmpctIdxs[PrefixSum - 1] = SrcIdx;
			});
		return dCmpctIdxs;
	}

	template <typename IdxTy, typename T>
	thrust::device_vector<T> Compact(const thrust::device_vector<T>& dSrcs,
		const thrust::device_vector<IdxTy>& dValids, IdxTy SrcNum = 0, cudaStream_t Stream = 0)
	{
		auto dCmpctIdxs =
			CompactIndexes(SrcNum == 0 ? static_cast<IdxTy>(dSrcs.size()) : SrcNum, dValids);

		thrust::device_vector<T> dCmpcts(dCmpctIdxs.size());
		thrust::for_each(thrust::cuda::par_nosync.on(Stream),
			thrust::make_counting_iterator(IdxTy(0)),
			thrust::make_counting_iterator(static_cast<IdxTy>(dCmpcts.size())),
			[Cmpcts = thrust::raw_pointer_cast(dCmpcts.data()),
				CmpctIdxs = thrust::raw_pointer_cast(dCmpctIdxs.data()),
				Srcs = thrust::raw_pointer_cast(dSrcs.data())] __device__(IdxTy CmpctIdx) {
				Cmpcts[CmpctIdx] = Srcs[CmpctIdxs[CmpctIdx]];
			});
		return dCmpcts;
	}

} // namespace CUDA

#endif
