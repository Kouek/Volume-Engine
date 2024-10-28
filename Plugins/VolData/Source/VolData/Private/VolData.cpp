// Copyright Epic Games, Inc. All Rights Reserved.

#include "VolData.h"

#define LOCTEXT_NAMESPACE "FVolDataModule"

void FVolDataModule::StartupModule()
{
	// This code will execute after your module is loaded into memory; the exact timing is specified
	// in the .uplugin file per-module
	auto ShadersDir = FPaths::Combine(FPaths::ProjectPluginsDir(), TEXT("VolData/Shaders"));
	AddShaderSourceDirectoryMapping(TEXT("/VolData"), ShadersDir);
}

void FVolDataModule::ShutdownModule()
{
	// This function may be called during shutdown to clean up your module.  For modules that
	// support dynamic reloading, we call this function before unloading the module.
}

#undef LOCTEXT_NAMESPACE

IMPLEMENT_MODULE(FVolDataModule, VolData)