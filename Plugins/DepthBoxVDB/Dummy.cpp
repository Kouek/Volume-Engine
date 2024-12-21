#include <DepthBoxVDB/VolRenderer.h>

#include <algorithm>
#include <execution>
#include <iostream>
#include <fstream>
#include <format>

#include <array>
#include <vector>

int main()
{
	using namespace DepthBoxVDB::VolData;

	constexpr std::array Dim{ 1024, 1024, 1024 };
	constexpr uint32_t	 VoxelNum = static_cast<uint32_t>(Dim[0]) * Dim[1] * Dim[2];
	constexpr int		 BrickSize = 32;
	constexpr std::array BrickPerVolume{ Dim[0] / BrickSize, Dim[1] / BrickSize,
		Dim[2] / BrickSize };
	constexpr int		 FrameNum = 8;
	constexpr float		 MaxRadius = .707f * std::min({ Dim[0], Dim[1], Dim[2] });
	constexpr float		 RadiusStep = MaxRadius / FrameNum;

	std::vector<uint8_t> Data(Dim[0] * Dim[1] * Dim[2], 0);

	auto Construct = [&](const std::array<float, 2>& RadiusRange, const std::array<int, 3>& Coord,
						 uint8_t* OutScalarPtr) {
		constexpr std::array Centric{ Dim[0] / 2, Dim[1] / 2, Dim[2] / 2 };

		float Dist = [&]() {
			std::array<float, 3> Diff{ Coord[0] + .5f - Centric[0], Coord[1] + .5f - Centric[1],
				Coord[2] + .5f - Centric[2] };
			return std::sqrtf(Diff[0] * Diff[0] + Diff[1] * Diff[1] + Diff[2] * Diff[2]);
		}();

		if (Dist < RadiusRange[0] || Dist > RadiusRange[1])
		{
			*OutScalarPtr = 0;
			return;
		}

		*OutScalarPtr = static_cast<uint8_t>(glm::clamp(
			255.f * (Dist - RadiusRange[0]) / (RadiusRange[1] - RadiusRange[0]), 0.f, 255.f));
	};

	std::vector<uint32_t> Indices(VoxelNum);
	std::ranges::generate(Indices, [Index = uint32_t(0)]() mutable { return Index++; });
	std::array<float, 2> RadiusRange{ 0.f, RadiusStep };
	for (int f = 0; f < FrameNum; ++f)
	{
		std::for_each(std::execution::seq, Indices.begin(), Indices.end(), [&](uint32_t Index) {
			std::array<int, 3> Coord;
			uint32_t		   Tmp = static_cast<uint32_t>(Dim[0]) * Dim[1];
			Coord[2] = Index / Tmp;
			Tmp = Index - Coord[2] * Tmp;
			Coord[1] = Tmp / Dim[0];
			Coord[0] = Tmp - Coord[1] * Dim[0];

			Construct(RadiusRange, Coord, Data.data() + Index);
		});
		// Uncomment this line to create Spherical Shell, else create Sphere
		RadiusRange[0] += RadiusStep;
		RadiusRange[1] += RadiusStep;

		std::string Name =
			std::format("test_data_{}x{}x{}_uint8_{}.raw", Dim[0], Dim[1], Dim[2], f);
		std::ofstream os(Name, std::ios::binary);
		assert(os.is_open());

		os.write((const char*)Data.data(), sizeof(uint8_t) * Data.size());
		std::cout << std::format("Write {}.\n", Name);
	}

	return 0;
}
