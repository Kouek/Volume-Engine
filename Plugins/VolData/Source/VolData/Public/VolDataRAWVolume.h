#pragma once

#include "CoreMinimal.h"

#include "Engine/VolumeTexture.h"

#include "VolDataUtil.h"

class FVolDataRAWVolumeData
{
public:
	struct LoadFromFileParameters
	{
		using RetValueType = TArray<uint8>;

		EVolDataVoxelType VoxelType;
		FIntVector3		  VoxelPerVolume;
		FIntVector3		  AxisOrder;
		FFilePath		  SourcePath;
	};
	static TVariant<LoadFromFileParameters::RetValueType, FString> LoadFromFile(const LoadFromFileParameters& Params);

	struct CreateTextureParameters
	{
		using RetValueType = UVolumeTexture*;

		EVolDataVoxelType	 VoxelType;
		FIntVector3			 VoxelPerVolume;
		const TArray<uint8>& RAWVolumeData;
	};
	static TVariant<CreateTextureParameters::RetValueType, FString> CreateTexture(
		const CreateTextureParameters& Params);
};
