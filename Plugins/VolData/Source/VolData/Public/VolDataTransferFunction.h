#pragma once

#include "CoreMinimal.h"

#include <type_traits>

#include "VolDataUtil.h"

class FVolDataTransferFunction
{
public:
	template <bool bUseHalf> struct LoadFromFileParameters
	{
		struct RetValueTrait
		{
			auto operator()()
			{
				if constexpr (bUseHalf)
					return TArray<FFloat16>();
				else
					return TArray<float>();
			}
		};
		using RetValueType = std::invoke_result_t<RetValueTrait>;

		uint32	  Resolution;
		FFilePath SourcePath;
	};
	template <bool bUseHalf>
	static TVariant<typename LoadFromFileParameters<bUseHalf>::RetValueType, FString> LoadFromFile(
		const LoadFromFileParameters<bUseHalf>& Params);

	template <bool bUseHalf>
	static LoadFromFileParameters<bUseHalf>::RetValueType LerpFromPointsToFlatArray(
		const TMap<float, FVector4f>& Points, uint32 Resolution = 256);

	template <bool bUseHalf>
	static LoadFromFileParameters<bUseHalf>::RetValueType PreIntegrateFromFlatArray(
		const typename LoadFromFileParameters<bUseHalf>::RetValueType& Array, uint32 Resolution = 256);

	struct CreateTextureParameters
	{
		using RetValueType = UTexture2D*;

		uint32					Resolution;
		const TArray<FFloat16>& TFFlatArray;
	};
	static TVariant<CreateTextureParameters::RetValueType, FString> CreateTexture(
		const CreateTextureParameters& Params);
	static TVariant<CreateTextureParameters::RetValueType, FString> CreateTexturePreIntegrated(
		const CreateTextureParameters& Params);
};
