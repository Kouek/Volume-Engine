#ifndef PUBLIC_DEPTHBOXVDB_UTIL_H
#define PUBLIC_DEPTHBOXVDB_UTIL_H

namespace DepthBoxVDB
{

	struct Noncopyable
	{
		Noncopyable() = default;
		Noncopyable(const Noncopyable&) = delete;
		Noncopyable& operator=(const Noncopyable&) = delete;
	};

} // namespace DepthBoxVDB

#endif
