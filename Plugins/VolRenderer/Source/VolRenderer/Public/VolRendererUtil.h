#pragma once

#include <iostream>
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
			if (str().empty())
			{
				return std::stringbuf::sync();
			}

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

	class FStdOutputLinker
	{
	public:
		FStdOutputLinker()
		{
			PrevOutputStreamBuffer = std::cout.rdbuf();
			PrevErrorStreamBuffer = std::cerr.rdbuf();
			std::cout.set_rdbuf(&FStdStream<ELogVerbosity::Log>::Instance());
			std::cerr.set_rdbuf(&FStdStream<ELogVerbosity::Error>::Instance());
		}
		~FStdOutputLinker()
		{
			std::cout.flush();
			std::cerr.flush();

			std::cout.set_rdbuf(PrevOutputStreamBuffer);
			std::cerr.set_rdbuf(PrevErrorStreamBuffer);
		}

	private:
		std::streambuf* PrevOutputStreamBuffer = nullptr;
		std::streambuf* PrevErrorStreamBuffer = nullptr;
	};

} // namespace VolRenderer
