#include "VolData.h"

#include <thrust/device_vector.h>

std::unique_ptr<DepthBoxVDB::VolData::IVDBBuilder> DepthBoxVDB::VolData::IVDBBuilder::Create()
{
	return std::make_unique<VDBBuilderImpl>();
}

void DepthBoxVDB::VolData::VDBBuilderImpl::FullBuild(const FullBuildParameters& Params) {}
