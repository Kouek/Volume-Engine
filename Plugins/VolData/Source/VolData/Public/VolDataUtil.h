#pragma once

#include <iostream>
#include <sstream>

#include "CoreMinimal.h"

DECLARE_LOG_CATEGORY_EXTERN(LogVolData, Log, All);

UENUM()
enum class EVolDataVoxelType : uint8
{
	None = 0 UMETA(DisplayName = "None"),
	UInt8	UMETA(DisplayName = "Unsigned Int 8 bit"),
	UInt16	UMETA(DisplayName = "Unsigned Int 16 bit"),
	Float32 UMETA(DisplayName = "Float 32 bit")
};

namespace VolData
{
	inline EPixelFormat CastVoxelTypeToPixelFormat(EVolDataVoxelType VoxelType)
	{
		switch (VoxelType)
		{
			case EVolDataVoxelType::UInt8:
				return EPixelFormat::PF_R8_UINT;
			case EVolDataVoxelType::UInt16:
				return EPixelFormat::PF_R16_UINT;
			case EVolDataVoxelType::Float32:
				return EPixelFormat::PF_R32_FLOAT;
			default:
				check(false);
				return EPixelFormat::PF_Unknown;
		}
	}

	inline size_t SizeOfVoxelType(EVolDataVoxelType VoxelType)
	{
		switch (VoxelType)
		{
			case EVolDataVoxelType::UInt8:
				return sizeof(uint8);
			case EVolDataVoxelType::UInt16:
				return sizeof(uint16);
			case EVolDataVoxelType::Float32:
				return sizeof(float);
			default:
				check(false);
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

	template <ELogVerbosity::Type Verbosity> class FStdStream : public std::stringbuf
	{
	public:
		static FStdStream& Instance()
		{
			static FStdStream Stream;
			return Stream;
		}

	protected:
		int sync()
		{
			if (str().empty())
			{
				return std::stringbuf::sync();
			}

			if constexpr (Verbosity == ELogVerbosity::Error)
			{
				UE_LOG(LogVolData, Error, TEXT("%s"), *FString(str().c_str()));
			}
			else if constexpr (Verbosity == ELogVerbosity::Log)
			{
				UE_LOG(LogVolData, Log, TEXT("%s"), *FString(str().c_str()));
			}

			str("");
			return std::stringbuf::sync();
		}
	};

	class FStdOutputLinker
	{
	public:
		FStdOutputLinker()
		{
			PrevOutputStreamBuffer = std::cout.rdbuf();
			PrevErrorStreamBuffer = std::cerr.rdbuf();
			std::cout.set_rdbuf(&FStdStream<ELogVerbosity::Log>::Instance());
			std::cerr.set_rdbuf(&FStdStream<ELogVerbosity::Error>::Instance());
		}
		~FStdOutputLinker()
		{
			std::cout.flush();
			std::cerr.flush();

			std::cout.set_rdbuf(PrevOutputStreamBuffer);
			std::cerr.set_rdbuf(PrevErrorStreamBuffer);
		}

	private:
		std::streambuf* PrevOutputStreamBuffer = nullptr;
		std::streambuf* PrevErrorStreamBuffer = nullptr;
	};

} // namespace VolData

template <typename T>
concept FVolDataVoxelType = std::is_same_v<T, uint8> || std::is_same_v<T, uint16> || std::is_same_v<T, float>;
