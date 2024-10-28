// Copyright Epic Games, Inc. All Rights Reserved.

#include "VolRenderer.h"

#include <iostream>

#include "VolRendererUtil.h"

#define LOCTEXT_NAMESPACE "FVolRendererModule"

void FVolRendererModule::StartupModule()
{
	// This code will execute after your module is loaded into memory; the exact timing is specified
	// in the .uplugin file per-module
	auto ShadersDir = FPaths::Combine(FPaths::ProjectPluginsDir(), TEXT("VolRenderer/Shaders"));
	AddShaderSourceDirectoryMapping(TEXT("/VolRenderer"), ShadersDir);

	std::cout.set_rdbuf(&VolRenderer::FStdStream<ELogVerbosity::Type::Log>::Instance());
	std::cerr.set_rdbuf(&VolRenderer::FStdStream<ELogVerbosity::Type::Error>::Instance());
}

void FVolRendererModule::ShutdownModule()
{
	// This function may be called during shutdown to clean up your module.  For modules that
	// support dynamic reloading, we call this function before unloading the module.
}

#undef LOCTEXT_NAMESPACE

IMPLEMENT_MODULE(FVolRendererModule, VolRenderer)