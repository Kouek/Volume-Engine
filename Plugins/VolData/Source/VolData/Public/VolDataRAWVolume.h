#pragma once

#include "CoreMinimal.h"

#include "Engine/VolumeTexture.h"

#include "VolDataUtil.h"

class FVolDataRAWVolumeData
{
public:
	struct LoadFromFileParameters
	{
		EVolDataVoxelType VoxelType;
		FIntVector3		  VoxelPerVolume;
		FIntVector3		  AxisOrder;
		FFilePath		  SourcePath;
	};
	static TVariant<TArray<uint8>, FString> LoadFromFile(const LoadFromFileParameters& Params);

	struct CreateTextureParameters
	{
		EVolDataVoxelType	 VoxelType;
		FIntVector3			 VoxelPerVolume;
		const TArray<uint8>& RAWVolumeData;
	};
	static TVariant<UVolumeTexture*, FString> CreateTexture(const CreateTextureParameters& Params);
};
