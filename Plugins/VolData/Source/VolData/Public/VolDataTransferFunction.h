#pragma once

#include "CoreMinimal.h"

#include <type_traits>

#include "VolDataUtil.h"

class FVolDataTransferFunction
{
public:
	struct LoadFromFileParameters
	{
		using RetValueType = TMap<float, FVector4f>;

		FFilePath SourcePath;
	};
	static TVariant<typename LoadFromFileParameters::RetValueType, FString> LoadFromFile(
		const LoadFromFileParameters& Params);

	template <bool bUseHalf> struct FlattenDataTrait
	{
		auto operator()()
		{
			if constexpr (bUseHalf)
				return TArray<FFloat16>();
			else
				return TArray<float>();
		}

		using Type = std::invoke_result_t<FlattenDataTrait>;
	};

	template <bool bUseHalf>
	static FlattenDataTrait<bUseHalf>::Type PreIntegrateFromFlatArray(
		const typename FlattenDataTrait<bUseHalf>::Type& Array, uint32 Resolution = 256);

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
