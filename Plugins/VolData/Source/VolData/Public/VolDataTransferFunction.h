#pragma once

#include "CoreMinimal.h"

#include "VolDataUtil.h"

class FVolDataTransferFunction
{
public:
	struct LoadFromFileParameters
	{
		uint32	  Resolution;
		FFilePath SourcePath;
	};
	static TVariant<TArray<float>, FString> LoadFromFile(const LoadFromFileParameters& Params);

	static TArray<float> LerpFromPointsToFlatArray(const TMap<float, FVector4f>& Points, uint32 Resolution = 256);
};
