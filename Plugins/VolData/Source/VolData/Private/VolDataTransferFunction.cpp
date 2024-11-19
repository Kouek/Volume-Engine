#include "VolDataTransferFunction.h"

#include <array>

TVariant<typename FVolDataTransferFunction::LoadFromFileParameters::RetValueType, FString>
FVolDataTransferFunction::LoadFromFile(const LoadFromFileParameters& Params)
{
	using RetType = TVariant<typename LoadFromFileParameters::RetValueType, FString>;

	FJsonSerializableArray Buffer;
	if (!FFileHelper::LoadANSITextFileToStrings(*Params.SourcePath.FilePath, nullptr, Buffer))
		return RetType(
			TInPlaceType<FString>(), FString::Format(TEXT("Invalid SourcePath {0}."), { Params.SourcePath.FilePath }));

	typename LoadFromFileParameters::RetValueType Points;
	for (int Line = 0; Line < Buffer.Num(); ++Line)
	{
		if (Buffer[Line].IsEmpty())
			continue;

		float LineVars[5] = { 0.f };
		int	  ValidCount = sscanf(StringCast<ANSICHAR>(*Buffer[Line]).Get(), "%f%f%f%f%f", &LineVars[0], &LineVars[1],
			  &LineVars[2], &LineVars[3], &LineVars[4]);
		if (ValidCount != 5)
			return RetType(TInPlaceType<FString>(),
				FString::Format(
					TEXT("Invalid contents at line {0} in SourcePath {1}."), { Line + 1, Params.SourcePath.FilePath }));

		auto& RGBA = Points.Emplace(LineVars[0]);
		for (int CmpIdx = 0; CmpIdx < 4; ++CmpIdx)
			RGBA[CmpIdx] = std::min(std::max(LineVars[CmpIdx + 1] / 255.f, 0.f), 1.f);
	}

	return RetType(TInPlaceType<typename LoadFromFileParameters::RetValueType>(), Points);
}

template <bool bUseHalf>
FVolDataTransferFunction::FlattenDataTrait<bUseHalf>::Type FVolDataTransferFunction::PreIntegrateFromFlatArray(
	const typename FlattenDataTrait<bUseHalf>::Type& Array, uint32 Resolution)
{
	using ElementType = typename FlattenDataTrait<bUseHalf>::Type::ElementType;

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

	typename FlattenDataTrait<bUseHalf>::Type DatPreInt;
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
template FVolDataTransferFunction::FlattenDataTrait<false>::Type
FVolDataTransferFunction::PreIntegrateFromFlatArray<false>(
	typename const FlattenDataTrait<false>::Type& Array, uint32 Resolution);
template FVolDataTransferFunction::FlattenDataTrait<true>::Type
FVolDataTransferFunction::PreIntegrateFromFlatArray<true>(
	typename const FlattenDataTrait<true>::Type& Array, uint32 Resolution);

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
