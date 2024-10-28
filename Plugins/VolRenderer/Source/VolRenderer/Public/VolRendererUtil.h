#pragma once

#include <sstream>

#include "CoreMinimal.h"
#include "Logging/LogCategory.h"

DECLARE_LOG_CATEGORY_EXTERN(LogVolRenderer, Log, All);

namespace VolRenderer
{
	template <ELogVerbosity::Type Verbosity> class FStdStream : public std::stringbuf
	{
	public:
		static FStdStream& Instance()
		{
			static FStdStream Stream;
			return Stream;
		}

	protected:
		int sync()
		{
			if constexpr (Verbosity == ELogVerbosity::Error)
			{
				UE_LOG(LogVolRenderer, Error, TEXT("%s"), *FString(str().c_str()));
			}
			else if constexpr (Verbosity == ELogVerbosity::Log)
			{
				UE_LOG(LogVolRenderer, Log, TEXT("%s"), *FString(str().c_str()));
			}

			str("");
			return std::stringbuf::sync();
		}
	};
} // namespace VolRenderer
