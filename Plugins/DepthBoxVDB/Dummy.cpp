#include <DepthBoxVDB/VolRenderer.h>

#include <iostream>
#include <fstream>
#include <format>

#include <array>
#include <vector>

int main()
{
	using namespace DepthBoxVDB::VolData;

	constexpr std::array Dim{ 256, 256, 256 };
	constexpr int		 BrickSize = 8;
	constexpr std::array BrickPerVolume{ Dim[0] / BrickSize, Dim[1] / BrickSize,
		Dim[2] / BrickSize };

	std::vector<uint8_t> Data(Dim[0] * Dim[1] * Dim[2], 0);
	uint8_t*			 Ptr = Data.data();
	for (int z = 0; z < Dim[2]; ++z)
		for (int y = 0; y < Dim[1]; ++y)
			for (int x = 0; x < Dim[0]; ++x, ++Ptr)
			{
				int bZ = z / BrickSize;
				int bY = y / BrickSize;
				int bX = x / BrickSize;
				if (bX == bY && bY == bZ)
					*Ptr = std::floor(255.f * std::max(1.f * bX / BrickPerVolume[0], .1f));
			}

	std::string	  Name = std::format("test_data_{}x{}x{}_uint8.raw", Dim[0], Dim[1], Dim[2]);
	std::ofstream os(Name, std::ios::binary);
	assert(os.is_open());
	os.write((const char*)Data.data(), sizeof(uint8_t) * Data.size());

	return 0;
}
