#include "VolDataTransferFunction.h"

TVariant<TArray<float>, FString> FVolDataTransferFunction::LoadFromFile(const LoadFromFileParameters& Params)
{
	using ValueType = TArray<float>;
	using RetType = TVariant<TArray<float>, FString>;

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
		auto  ValidCount = sscanf(StringCast<ANSICHAR>(*Buffer[Line]).Get(), "%f%f%f%f%f", &LineVars[0], &LineVars[1],
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

	return RetType(TInPlaceType<ValueType>(), LerpFromPointsToFlatArray(Points, Params.Resolution));
}

TArray<float> FVolDataTransferFunction::LerpFromPointsToFlatArray(
	const TMap<float, FVector4f>& Points, uint32 Resolution)
{
	TArray<float> Data;

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
