cmake_minimum_required(VERSION 3.26)

# --- Client configuration ---
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/client_config.cmake")
    include(client_config.cmake)

    if (NOT DIGITALCURLING_CLIENT_NAME)
        message(FATAL_ERROR "DIGITALCURLING_CLIENT_NAME is not set in client_config.cmake.")
    endif()
    if (NOT DIGITALCURLING_CLIENT_VERSION)
        message(FATAL_ERROR "DIGITALCURLING_CLIENT_VERSION is not set in client_config.cmake.")
    endif()
else()
    message(FATAL_ERROR "client_config.cmake not found.")
endif()

# --- Project settings ---
project("${DIGITALCURLING_CLIENT_NAME}_v${DIGITALCURLING_CLIENT_VERSION}"
    VERSION ${DIGITALCURLING_CLIENT_VERSION}
    LANGUAGES CXX
)

# --- Build settings ---
# Search strength depends on speed, so a plain configure must not build unoptimised.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "No build type given; using Release")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_compile_options($<$<CXX_COMPILER_ID:MSVC>:/utf-8>)

# --- Build options ---
option(DIGITALCURLING_CLIENT_BUILD_STANDARD_CLIENT "Enable support for standard client" ON)
option(DIGITALCURLING_CLIENT_BUILD_MIXED_CLIENT "Enable support for mixed client" ON)
option(DIGITALCURLING_CLIENT_BUILD_MIXED_DOUBLES_CLIENT "Enable support for mixed doubles client" ON)
option(DIGITALCURLING_CLIENT_BUILD_TESTS "Build unit tests (GoogleTest)" OFF)

if (DIGITALCURLING_CLIENT_BUILD_TESTS)
    enable_testing()
endif()

# --- Build external libraries ---
# renovate: datasource=github-tags depName=pon-pokapoka/DigitalCurling extractVersion=^v(?<version>.+)$
set(DIGITALCURLING_CLIENT_DCLIB_VERSION "4.0.2")
# renovate: datasource=github-tags depName=CLIUtils/CLI11 extractVersion=^v(?<version>.+)$
set(DIGITALCURLING_CLIENT_CLILIB_VERSION "2.6.1")
# renovate: datasource=github-tags depName=yhirose/cpp-httplib extractVersion=^v(?<version>.+)$
set(DIGITALCURLING_CLIENT_HTTPLIB_VERSION "0.29.0")

# Digital Curling
include(FetchContent)
FetchContent_Declare(
    digitalcurling
    GIT_REPOSITORY https://github.com/pon-pokapoka/DigitalCurling.git
    GIT_TAG        v${DIGITALCURLING_CLIENT_DCLIB_VERSION}
)
set(DIGITALCURLING_BUILD_CORE ON)
set(DIGITALCURLING_BUILD_PLAYERS ${DIGITALCURLING_CLIENT_USE_LOADER})
set(DIGITALCURLING_BUILD_SIMULATORS ${DIGITALCURLING_CLIENT_USE_LOADER})
set(DIGITALCURLING_BUILD_PLUGIN_LOADER ${DIGITALCURLING_CLIENT_USE_LOADER})
set(DIGITALCURLING_BUNDLE_PLUGINS ${DIGITALCURLING_CLIENT_USE_LOADER})
FetchContent_MakeAvailable(digitalcurling)

# command line parser
FetchContent_Declare(
    cli11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11
    GIT_TAG        v${DIGITALCURLING_CLIENT_CLILIB_VERSION}
)
FetchContent_MakeAvailable(cli11)

# HTTP client / OpenSSL
find_package(OpenSSL QUIET)
if (OpenSSL_FOUND)
    set(HTTPLIB_REQUIRE_OPENSSL ON)
else()
    add_compile_definitions(DIGITALCURLING_CLIENT_NO_SSL)
endif()

FetchContent_Declare(
    httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib/
    GIT_TAG        v${DIGITALCURLING_CLIENT_HTTPLIB_VERSION}
)
set(HTTPLIB_USE_ZSTD_IF_AVAILABLE OFF)
FetchContent_MakeAvailable(httplib)

# --- Build client ---
add_subdirectory(src)
