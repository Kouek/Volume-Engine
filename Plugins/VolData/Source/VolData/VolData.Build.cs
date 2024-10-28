// Copyright Epic Games, Inc. All Rights Reserved.

using Microsoft.Extensions.Logging;
using System.IO;
using System.Linq;
using UnrealBuildTool;

public class VolData : ModuleRules
{
    public VolData(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicIncludePaths.AddRange(
            new string[] {
				// ... add public include paths required here ...
			}
            );


        PrivateIncludePaths.AddRange(
            new string[] {
				// ... add other private include paths required here ...
			}
            );


        PublicDependencyModuleNames.AddRange(
            new string[]
            {
                "Core",
                "RenderCore",
                "RHI"
				// ... add other public dependencies that you statically link with here ...
            }
            );


        PrivateDependencyModuleNames.AddRange(
            new string[]
            {
                "CoreUObject",
                "Engine",
                "Slate",
                "SlateCore",
				// ... add private dependencies that you statically link with here ...	
			}
            );


        DynamicallyLoadedModuleNames.AddRange(
            new string[]
            {
				// ... add any modules that your module loads dynamically here ...
			}
            );

        CppStandard = CppStandardVersion.Cpp20;

        LinkCUDA();
        LinkDepthBoxVDB();
    }

    private void LinkCUDA()
    {
        var CUDAPath = System.Environment.GetEnvironmentVariable("CUDA_PATH");
        var IncDir = Path.Combine(CUDAPath, "include");
        var LibDir = Path.Combine(CUDAPath, "lib/x64");

        PublicIncludePaths.Add(IncDir);
        PublicAdditionalLibraries.Add(Path.Combine(LibDir, "cudart_static.lib"));

        Logger.LogInformation("Add CUDA Inc: {}", IncDir);
    }

    private void LinkDepthBoxVDB()
    {
        string ConfigDir = "";
        switch (Target.Configuration)
        {
            case UnrealTargetConfiguration.Shipping:
                ConfigDir = "Release"; break;
            default:
                ConfigDir = "RelWithDebInfo"; break;
        }
        string TargetDir = "DepthBoxVDB";

        var PluginsDir = Path.Combine(PluginDirectory, "../");
        var LibDir = Path.Combine(PluginsDir, TargetDir, "Binaries");
        LibDir = Path.Combine(LibDir, ConfigDir);
        var IncDir = Path.Combine(PluginsDir, TargetDir, "Source/Public");

        var GLMIncDir = Path.Combine(PluginsDir, TargetDir, "ThirdParty/glm");

        PublicAdditionalLibraries.Add(Path.Combine(LibDir, TargetDir + ".lib"));
        PublicIncludePaths.AddRange(new string[]{
            IncDir,
            GLMIncDir
        });

        Logger.LogInformation("Add DepthBoxVDB Inc: {}", PublicIncludePaths);
    }
}
