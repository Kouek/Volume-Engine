#include "VolDataTransferFunction.h"

#include <array>

template <bool bUseHalf>
TVariant<typename FVolDataTransferFunction::LoadFromFileParameters<bUseHalf>::RetValueType, FString>
FVolDataTransferFunction::LoadFromFile(const LoadFromFileParameters<bUseHalf>& Params)
{
	using RetType = TVariant<typename LoadFromFileParameters<bUseHalf>::RetValueType, FString>;

	FJsonSerializableArray Buffer;
	if (!FFileHelper::LoadANSITextFileToStrings(*Params.SourcePath.FilePath, nullptr, Buffer))
		return RetType(
			TInPlaceType<FString>(), FString::Format(TEXT("Invalid SourcePath {0}."), { Params.SourcePath.FilePath }));

	TMap<float, FVector4f> Points;
	for (int Line = 0; Line < Buffer.Num(); ++Line)
	{
		if (Buffer[Line].IsEmpty())
			continue;

		float LineVars[5] = { 0.f };
		int	  ValidCount = sscanf(StringCast<ANSICHAR>(*Buffer[Line]).Get(), "%f%f%f%f%f", &LineVars[0], &LineVars[1],
			  &LineVars[2], &LineVars[3], &LineVars[4]);
		if (ValidCount != 5 || [&]() {
				for (int i = 0; i < 5; ++i)
					if (LineVars[i] < 0.f || LineVars[i] >= 255.5f)
						return true;
				return false;
			}())
			return RetType(TInPlaceType<FString>(),
				FString::Format(
					TEXT("Invalid contents at line {0} in SourcePath {1}."), { Line + 1, Params.SourcePath.FilePath }));

		auto& RGBA = Points.Emplace(LineVars[0]);
		for (int CmpIdx = 0; CmpIdx < 4; ++CmpIdx)
			RGBA[CmpIdx] = std::min(std::max(LineVars[CmpIdx + 1] / 255.f, 0.f), 1.f);
	}

	return RetType(TInPlaceType<typename LoadFromFileParameters<bUseHalf>::RetValueType>(),
		LerpFromPointsToFlatArray<bUseHalf>(Points, Params.Resolution));
}
template TVariant<typename FVolDataTransferFunction::LoadFromFileParameters<false>::RetValueType, FString>
FVolDataTransferFunction::LoadFromFile<false>(const LoadFromFileParameters<false>& Params);
template TVariant<typename FVolDataTransferFunction::LoadFromFileParameters<true>::RetValueType, FString>
FVolDataTransferFunction::LoadFromFile<true>(const LoadFromFileParameters<true>& Params);

template <bool bUseHalf>
FVolDataTransferFunction::LoadFromFileParameters<bUseHalf>::RetValueType
FVolDataTransferFunction::LerpFromPointsToFlatArray(const TMap<float, FVector4f>& Points, uint32 Resolution)
{
	typename LoadFromFileParameters<bUseHalf>::RetValueType Data;

	Data.Reserve(Resolution * 4);

	auto Itr = Points.begin();
	auto ItrPrev = Itr;
	++Itr;
	for (uint32 Scalar = 0; Scalar < Resolution; ++Scalar)
	{
		if (Scalar > Itr->Key)
		{
			++ItrPrev;
			++Itr;
		}
		auto& [ScalarCurr, ColorCurr] = *Itr;
		auto& [ScalarPrev, ColorPrev] = *ItrPrev;

		auto K = ScalarCurr == ScalarPrev ? 1.f : (Scalar - ScalarPrev) / (ScalarCurr - ScalarPrev);
		auto RGBA = (1.f - K) * ColorPrev + K * ColorCurr;

		Data.Emplace(RGBA.X);
		Data.Emplace(RGBA.Y);
		Data.Emplace(RGBA.Z);
		Data.Emplace(RGBA.W);
	}

	return Data;
}
template FVolDataTransferFunction::LoadFromFileParameters<false>::RetValueType
FVolDataTransferFunction::LerpFromPointsToFlatArray<false>(const TMap<float, FVector4f>& Points, uint32 Resolution);
template FVolDataTransferFunction::LoadFromFileParameters<true>::RetValueType
FVolDataTransferFunction::LerpFromPointsToFlatArray<true>(const TMap<float, FVector4f>& Points, uint32 Resolution);

template <bool bUseHalf>
FVolDataTransferFunction::LoadFromFileParameters<bUseHalf>::RetValueType
FVolDataTransferFunction::PreIntegrateFromFlatArray(
	const typename LoadFromFileParameters<bUseHalf>::RetValueType& Array, uint32 Resolution)
{
	using ElementType = typename FVolDataTransferFunction::LoadFromFileParameters<bUseHalf>::RetValueType::ElementType;

	TArray<std::array<float, 4>>	  DatPreMultAlphaAvg;
	const std::array<ElementType, 4>* Dat = reinterpret_cast<const std::array<ElementType, 4>*>(Array.GetData());
	DatPreMultAlphaAvg.SetNum(Resolution);
	{
		DatPreMultAlphaAvg[0][0] = Dat[0][0];
		DatPreMultAlphaAvg[0][1] = Dat[0][1];
		DatPreMultAlphaAvg[0][2] = Dat[0][2];
		DatPreMultAlphaAvg[0][3] = Dat[0][3];
		for (uint32 i = 1; i < Resolution; ++i)
		{
			float a = .5f * (Dat[i - 1][3] + Dat[i][3]);
			float r = .5f * (Dat[i - 1][0] + Dat[i][0]) * a;
			float g = .5f * (Dat[i - 1][1] + Dat[i][1]) * a;
			float b = .5f * (Dat[i - 1][2] + Dat[i][2]) * a;

			DatPreMultAlphaAvg[i][0] = DatPreMultAlphaAvg[i - 1][0] + r;
			DatPreMultAlphaAvg[i][1] = DatPreMultAlphaAvg[i - 1][1] + g;
			DatPreMultAlphaAvg[i][2] = DatPreMultAlphaAvg[i - 1][2] + b;
			DatPreMultAlphaAvg[i][3] = DatPreMultAlphaAvg[i - 1][3] + a;
		}
	}

	typename FVolDataTransferFunction::LoadFromFileParameters<bUseHalf>::RetValueType DatPreInt;
	DatPreInt.SetNum(4 * Resolution * Resolution);

	std::array<ElementType, 4>* DatPtrPreInt = reinterpret_cast<std::array<ElementType, 4>*>(DatPreInt.GetData());
	for (uint32 ScalarFront = 0; ScalarFront < Resolution; ++ScalarFront)
		for (uint32 ScalarBack = 0; ScalarBack < Resolution; ++ScalarBack)
		{
			auto ScalarMin = ScalarFront;
			auto ScalarMax = ScalarBack;
			if (ScalarFront > ScalarBack)
				std::swap(ScalarMin, ScalarMax);

			if (ScalarMin == ScalarMax)
			{
				float a = Dat[ScalarMin][3];
				(*DatPtrPreInt)[0] = Dat[ScalarMin][0] * a;
				(*DatPtrPreInt)[1] = Dat[ScalarMin][1] * a;
				(*DatPtrPreInt)[2] = Dat[ScalarMin][2] * a;
				(*DatPtrPreInt)[3] = 1.f - FMath::Exp(-a);
			}
			else
			{
				auto factor = 1.f / (ScalarMax - ScalarMin);
				(*DatPtrPreInt)[0] = (DatPreMultAlphaAvg[ScalarMax][0] - DatPreMultAlphaAvg[ScalarMin][0]) * factor;
				(*DatPtrPreInt)[1] = (DatPreMultAlphaAvg[ScalarMax][1] - DatPreMultAlphaAvg[ScalarMin][1]) * factor;
				(*DatPtrPreInt)[2] = (DatPreMultAlphaAvg[ScalarMax][2] - DatPreMultAlphaAvg[ScalarMin][2]) * factor;
				(*DatPtrPreInt)[3] =
					1.f - FMath::Exp((DatPreMultAlphaAvg[ScalarMin][3] - DatPreMultAlphaAvg[ScalarMax][3]) * factor);
			}

			++DatPtrPreInt;
		}

	return DatPreInt;
}
template FVolDataTransferFunction::LoadFromFileParameters<false>::RetValueType
FVolDataTransferFunction::PreIntegrateFromFlatArray<false>(
	typename const LoadFromFileParameters<false>::RetValueType& Array, uint32 Resolution);
template FVolDataTransferFunction::LoadFromFileParameters<true>::RetValueType
FVolDataTransferFunction::PreIntegrateFromFlatArray<true>(
	typename const LoadFromFileParameters<true>::RetValueType& Array, uint32 Resolution);

TVariant<FVolDataTransferFunction::CreateTextureParameters::RetValueType, FString>
FVolDataTransferFunction::CreateTexture(const CreateTextureParameters& Params)
{
	using RetType = TVariant<CreateTextureParameters::RetValueType, FString>;

	if (Params.TFFlatArray.Num() != 4 * Params.Resolution)
	{
		return RetType(TInPlaceType<FString>(),
			FString::Format(
				TEXT("TFFlatArray:{0} != 4 * Resolution:{1}."), { Params.TFFlatArray.Num(), Params.Resolution }));
	}

	auto Tex = UTexture2D::CreateTransient(Params.Resolution, 1, PF_FloatRGBA);

	Tex->Filter = TextureFilter::TF_Bilinear;
	Tex->AddressX = Tex->AddressY = TextureAddress::TA_Clamp;

	void* TexDat = Tex->GetPlatformData()->Mips[0].BulkData.Lock(EBulkDataLockFlags::LOCK_READ_WRITE);
	FMemory::Memmove(TexDat, Params.TFFlatArray.GetData(), sizeof(FFloat16) * 4 * Params.Resolution);
	Tex->GetPlatformData()->Mips[0].BulkData.Unlock();
	Tex->UpdateResource();

	return RetType(TInPlaceType<CreateTextureParameters::RetValueType>(), Tex);
}

TVariant<FVolDataTransferFunction::CreateTextureParameters::RetValueType, FString>
FVolDataTransferFunction::CreateTexturePreIntegrated(const CreateTextureParameters& Params)
{
	using RetType = TVariant<CreateTextureParameters::RetValueType, FString>;

	if (Params.TFFlatArray.Num() != 4 * Params.Resolution)
	{
		return RetType(TInPlaceType<FString>(),
			FString::Format(
				TEXT("TFFlatArray:{0} != 4 * Resolution:{1}."), { Params.TFFlatArray.Num(), Params.Resolution }));
	}

	auto Tex = UTexture2D::CreateTransient(Params.Resolution, Params.Resolution, PF_FloatRGBA);
	Tex->Filter = TextureFilter::TF_Bilinear;
	Tex->AddressX = Tex->AddressY = TextureAddress::TA_Clamp;

	void* TexDat = Tex->GetPlatformData()->Mips[0].BulkData.Lock(EBulkDataLockFlags::LOCK_READ_WRITE);
	auto  DataPreInt = PreIntegrateFromFlatArray<true>(Params.TFFlatArray, Params.Resolution);
	FMemory::Memmove(TexDat, DataPreInt.GetData(), sizeof(FFloat16) * 4 * Params.Resolution * Params.Resolution);
	Tex->GetPlatformData()->Mips[0].BulkData.Unlock();
	Tex->UpdateResource();

	return RetType(TInPlaceType<CreateTextureParameters::RetValueType>(), Tex);
}
