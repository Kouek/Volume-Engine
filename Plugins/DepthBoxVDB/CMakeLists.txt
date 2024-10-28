cmake_minimum_required(VERSION 3.18)

project("DepthBoxVDB" LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/Binaries")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/Binaries")

set(THIRD_PARTY_DIR "${CMAKE_CURRENT_LIST_DIR}/ThirdParty")

#<dep>
#<glm>
add_subdirectory("${THIRD_PARTY_DIR}/glm")
#<\glm>
#<\dep>

# <lib>
# <DepthBoxVDB>
set(TARGET_NAME "DepthBoxVDB")
add_library(
	${TARGET_NAME}
	STATIC
	"${CMAKE_CURRENT_LIST_DIR}/Source/Private/${TARGET_NAME}/VolData.cu"
	"${CMAKE_CURRENT_LIST_DIR}/Source/Private/${TARGET_NAME}/VolRenderer.cu"
)
set(LIBS "glm::glm")
target_link_libraries(
	${TARGET_NAME}
	PUBLIC
	${LIBS}
)
set(
	INCS
	"${CMAKE_CURRENT_LIST_DIR}/Source/Public"
	${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} # For IntelliSense
)
target_include_directories(
	${TARGET_NAME}
	PUBLIC
	${INCS}
)
target_compile_options(
	${TARGET_NAME}
	PRIVATE
	$<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)
# </DepthBoxVDB>
# </lib>

#<app>
#<Dummy>
set(TARGET_NAME "VolDataDummy")
add_executable(
	${TARGET_NAME}
	"${CMAKE_CURRENT_LIST_DIR}/Dummy.cpp"
)

target_link_libraries(
	${TARGET_NAME}
	PRIVATE
	"DepthBoxVDB"
)
#</Dummy>
#</app>
