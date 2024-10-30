#include "VolDataRAWVolume.h"

TVariant<TArray<uint8>, FString> FVolDataRAWVolumeData::LoadFromFile(const LoadFromFileParameters& Params)
{
	using RetType = TVariant<TArray<uint8>, FString>;

	if (Params.VoxelPerVolume.X <= 0 || Params.VoxelPerVolume.Y <= 0 || Params.VoxelPerVolume.Z <= 0)
		return RetType(TInPlaceType<FString>(),
			FString::Format(TEXT("Invalid VoxelPerVolume {0}."), { Params.VoxelPerVolume.ToString() }));

	TArray<uint8> Buf;
	if (!FFileHelper::LoadFileToArray(Buf, *Params.SourcePath.FilePath))
		return RetType(
			TInPlaceType<FString>(), FString::Format(TEXT("Invalid SourcePath {0}."), { Params.SourcePath.FilePath }));

	auto Transform = [&]<FVolDataVoxelType T>(T*) -> TOptional<FString> {
		auto ReorderOpt = VolData::ReorderVoxelPerVolume(Params.VoxelPerVolume, Params.AxisOrder);
		if (!ReorderOpt.IsSet())
			return FString::Format(TEXT("Invalid AxisOrder {0}."), { Params.AxisOrder.ToString() });

		auto [AxisOrderMap, VoxelPerVolume] = ReorderOpt.GetValue();

		T*			  TypedBuf = reinterpret_cast<T*>(Buf.GetData());
		decltype(Buf) OldBuf = Buf;
		T*			  TypedOldBuf = reinterpret_cast<T*>(OldBuf.GetData());
		size_t		  OldBufOffs = 0;
		size_t		  TrVoxYxX = static_cast<size_t>(VoxelPerVolume.Y) * VoxelPerVolume.X;
		FIntVector3	  Coord;
		for (Coord.Z = 0; Coord.Z < Params.VoxelPerVolume.Z; ++Coord.Z)
			for (Coord.Y = 0; Coord.Y < Params.VoxelPerVolume.Y; ++Coord.Y)
				for (Coord.X = 0; Coord.X < Params.VoxelPerVolume.X; ++Coord.X)
				{
					FIntVector3 trCoord(
						Params.AxisOrder.X > 0 ? Coord[AxisOrderMap[0]] : VoxelPerVolume.X - 1 - Coord[AxisOrderMap[0]],
						Params.AxisOrder.Y > 0 ? Coord[AxisOrderMap[1]] : VoxelPerVolume.Y - 1 - Coord[AxisOrderMap[1]],
						Params.AxisOrder.Z > 0 ? Coord[AxisOrderMap[2]]
											   : VoxelPerVolume.Z - 1 - Coord[AxisOrderMap[2]]);
					TypedBuf[trCoord.Z * TrVoxYxX + trCoord.Y * VoxelPerVolume.X + trCoord.X] = TypedOldBuf[OldBufOffs];
					++OldBufOffs;
				}

		return {};
	};

	auto Load = [&]<FVolDataVoxelType T>(T* Dummy) -> RetType {
		size_t VolSz = sizeof(T) * Params.VoxelPerVolume.X * Params.VoxelPerVolume.Y * Params.VoxelPerVolume.Z;

		if (Buf.Num() != VolSz)
			return RetType(TInPlaceType<FString>(),
				FString::Format(TEXT("Invalid contents in SourcePath {0}."), { Params.SourcePath.FilePath }));

		if (auto ErrMsgOpt = Transform(Dummy); ErrMsgOpt.IsSet())
			return RetType(TInPlaceType<FString>(), ErrMsgOpt.GetValue());

		return RetType(TInPlaceType<decltype(Buf)>(), Buf);
	};

	switch (Params.VoxelType)
	{
		case EVolDataVoxelType::UInt8:
			return Load((uint8*)nullptr);
		case EVolDataVoxelType::Float32:
			return Load((float*)nullptr);
		default:
			return RetType(TInPlaceType<FString>(), TEXT("Invalid VoxelType."));
	}
}

TVariant<UVolumeTexture*, FString> FVolDataRAWVolumeData::CreateTexture(const CreateTextureParameters& Params)
{
	using RetType = TVariant<UVolumeTexture*, FString>;

	auto Create = [&]<FVolDataVoxelType T>(T* Dummy) -> RetType {
		size_t VolSz = sizeof(T) * Params.VoxelPerVolume.X * Params.VoxelPerVolume.Y * Params.VoxelPerVolume.Z;
		if (Params.RAWVolumeData.Num() != VolSz)
			return RetType(TInPlaceType<FString>(), TEXT("Invalid contents in RAWVolumeData."));

		EPixelFormat PixFmt = VolData::CastVoxelTypeToPixelFormat(Params.VoxelType);
		auto		 Tex = UVolumeTexture::CreateTransient(
			Params.VoxelPerVolume.X, Params.VoxelPerVolume.Y, Params.VoxelPerVolume.Z, PixFmt);

		auto* TexDat = Tex->GetPlatformData()->Mips[0].BulkData.Lock(EBulkDataLockFlags::LOCK_READ_WRITE);
		FMemory::Memcpy(TexDat, Params.RAWVolumeData.GetData(), VolSz);
		Tex->GetPlatformData()->Mips[0].BulkData.Unlock();

		Tex->UpdateResource();

		return RetType(TInPlaceType<UVolumeTexture*>(), Tex);
	};

	switch (Params.VoxelType)
	{
		case EVolDataVoxelType::UInt8:
			return Create((uint8*)nullptr);
		case EVolDataVoxelType::Float32:
			return Create((float*)nullptr);
		default:
			return RetType(TInPlaceType<FString>(), TEXT("Invalid VoxelType."));
	}
}
