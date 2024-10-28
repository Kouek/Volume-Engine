#pragma once

#include "CoreMinimal.h"

#include "Engine/VolumeTexture.h"

UENUM()
enum class EVolDataVoxelType : uint8
{
	None = 0 UMETA(DisplayName = "None"),
	UInt8	UMETA(DisplayName = "Unsigned Int 8 bit"),
	Float32 UMETA(DisplayName = "Float 32 bit"),
	MAX
};

namespace VolData
{
	inline EPixelFormat CastVoxelTypeToPixelFormat(EVolDataVoxelType VoxelType)
	{
		switch (VoxelType)
		{
			case EVolDataVoxelType::UInt8:
				return EPixelFormat::PF_R8;
			case EVolDataVoxelType::Float32:
				return EPixelFormat::PF_R32_FLOAT;
			default:
				return EPixelFormat::PF_Unknown;
		}
	}

	inline size_t SizeOfVoxelType(EVolDataVoxelType VoxelType)
	{
		switch (VoxelType)
		{
			case EVolDataVoxelType::UInt8:
				return sizeof(uint8);
			case EVolDataVoxelType::Float32:
				return sizeof(float);
			default:
				return 0;
		}
	}

	inline TOptional<TTuple<FIntVector3, FIntVector3>> ReorderVoxelPerVolume(
		FIntVector3 VoxelPerVolume, FIntVector3 AxisOrder)
	{
		if ([&]() {
				FIntVector3 Cnts(0, 0, 0);
				for (int i = 0; i < 3; ++i)
					if (auto j = std::abs(AxisOrder[i]) - 1; 0 <= j && j <= 2)
						++Cnts[j];
					else
						return true;
				for (int i = 0; i < 3; ++i)
					if (Cnts[i] != 1)
						return true;
				return false;
			}())
			return {};

		FIntVector3 AxisOrderMap(std::abs(AxisOrder[0]) - 1, std::abs(AxisOrder[1]) - 1, std::abs(AxisOrder[2]) - 1);
		FIntVector3 ReorderedVoxelPerVolume(
			VoxelPerVolume[AxisOrderMap[0]], VoxelPerVolume[AxisOrderMap[1]], VoxelPerVolume[AxisOrderMap[2]]);
		return MakeTuple(AxisOrderMap, ReorderedVoxelPerVolume);
	}

} // namespace VolData

template <typename T>
concept FVolDataVoxelType = std::is_same_v<T, uint8> || std::is_same_v<T, uint16> || std::is_same_v<T, float>;

DECLARE_LOG_CATEGORY_EXTERN(LogVolData, Log, All);

class FRAWVolumeData
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
