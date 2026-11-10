function(target_compile_definition_if_exists target_name definition variable_name)
    set(variable_value "${${variable_name}}")
    if(NOT variable_value STREQUAL "")
        target_compile_definitions(${target_name} PRIVATE ${definition}=${variable_value})
    endif()
endfunction()

# Torch (libtorch)
set(LIBTORCH_DIR /usr/local/libtorch CACHE PATH "LibTorch install prefix")
set(CAFFE2_USE_CUDNN 1)
set(CAFFE2_USE_CUSPARSELT 1)
list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_DIR})
find_package(Torch REQUIRED)

# OpenMP
find_package(OpenMP REQUIRED COMPONENTS CXX)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# --- Build engine library ---
# Engine sources (DIGITALCURLING_CLIENT_SOURCES from client_config.cmake) are
# compiled once into a static library shared by the client and the unit tests.
set(DIGITALCURLING_CLIENT_ENGINE_LIB ${PROJECT_NAME}_engine)
add_library(${DIGITALCURLING_CLIENT_ENGINE_LIB} STATIC ${DIGITALCURLING_CLIENT_SOURCES})
target_include_directories(${DIGITALCURLING_CLIENT_ENGINE_LIB}
    PUBLIC ${CMAKE_SOURCE_DIR}/include
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(${DIGITALCURLING_CLIENT_ENGINE_LIB} PUBLIC ${TORCH_LIBRARIES})
target_compile_features(${DIGITALCURLING_CLIENT_ENGINE_LIB} PUBLIC cxx_std_17)
# The engine only needs the header-only plugin API; the client executable links
# the plugin loader (and the bundled plugins) itself.
target_link_libraries(${DIGITALCURLING_CLIENT_ENGINE_LIB} PUBLIC digitalcurling::plugin_api)
if (WIN32)
    target_compile_definitions(${DIGITALCURLING_CLIENT_ENGINE_LIB} PRIVATE WIN32_LEAN_AND_MEAN)
endif()

# --- Build client ---
# sources
add_executable(${PROJECT_NAME}
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/client_setup.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/client/client_base.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/client/client_factory.cpp
)
target_include_directories(${PROJECT_NAME}
    PRIVATE ${CMAKE_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(${PROJECT_NAME} PRIVATE ${DIGITALCURLING_CLIENT_ENGINE_LIB} CLI11::CLI11 httplib ${TORCH_LIBRARIES})
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

# definitions
if (DIGITALCURLING_CLIENT_BUILD_STANDARD_CLIENT)
    target_sources(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/client/standard_client.cpp)
    target_compile_definitions(${PROJECT_NAME} PRIVATE DIGITALCURLING_CLIENT_BUILD_STANDARD_CLIENT)
endif()

if (DIGITALCURLING_CLIENT_BUILD_MIXED_CLIENT)
    target_sources(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/client/mixed_client.cpp)
    target_compile_definitions(${PROJECT_NAME} PRIVATE DIGITALCURLING_CLIENT_BUILD_MIXED_CLIENT)
endif()

if (DIGITALCURLING_CLIENT_BUILD_MIXED_DOUBLES_CLIENT)
    target_sources(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/client/mixed_doubles_client.cpp)
    target_compile_definitions(${PROJECT_NAME} PRIVATE DIGITALCURLING_CLIENT_BUILD_MIXED_DOUBLES_CLIENT)
endif()

target_compile_definitions(${PROJECT_NAME}
    PRIVATE DIGITALCURLING_CLIENT_NAME="${DIGITALCURLING_CLIENT_NAME}"
    PRIVATE DIGITALCURLING_CLIENT_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
)
target_compile_definition_if_exists(${PROJECT_NAME} DIGITALCURLING_CLIENT_VERSION_MINOR PROJECT_VERSION_MINOR)
target_compile_definition_if_exists(${PROJECT_NAME} DIGITALCURLING_CLIENT_VERSION_PATCH PROJECT_VERSION_PATCH)

# Build ID: which code the client was built from (src/build_id.cmake). Set
# DIGITALCURLING_CLIENT_BUILD_ID where there is no .git, as the Docker build does.
set(DIGITALCURLING_CLIENT_BUILD_ID "" CACHE STRING "Build ID to embed instead of git describe (empty: use git)")
add_custom_target(${PROJECT_NAME}_build_id
    COMMAND ${CMAKE_COMMAND}
        -DSOURCE_DIR=${PROJECT_SOURCE_DIR}
        -DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/build_id.hpp
        -DBUILD_ID_OVERRIDE=${DIGITALCURLING_CLIENT_BUILD_ID}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/build_id.cmake
    BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/build_id.hpp
    COMMENT "Updating build ID"
    VERBATIM
)
add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_build_id)

# dependencies
if (OpenSSL_FOUND)
    target_include_directories(${PROJECT_NAME} PRIVATE ${OPENSSL_INCLUDE_DIR})
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenSSL::SSL OpenSSL::Crypto)
else()
    target_compile_definitions(${PROJECT_NAME} PRIVATE DIGITALCURLING_CLIENT_NO_SSL)
endif()

if (WIN32)
    target_compile_definitions(${PROJECT_NAME} PRIVATE WIN32_LEAN_AND_MEAN)
else()
    target_link_libraries(${PROJECT_NAME} PRIVATE ${CMAKE_DL_LIBS})
endif()

if (DIGITALCURLING_CLIENT_USE_LOADER)
    target_compile_definitions(${PROJECT_NAME} PRIVATE DIGITALCURLING_CLIENT_USE_LOADER)
    target_link_libraries(${PROJECT_NAME} PUBLIC digitalcurling::plugin_loader)
    message(STATUS "Building client with plugin loader")
else()
    target_link_libraries(${PROJECT_NAME} PUBLIC digitalcurling::plugin_api)
    message(STATUS "Building client with no plugin loader")
endif()

# --- Unit tests ---
# Added from here (not the top-level CMakeLists.txt) so the tests see the Torch
# targets and OpenMP flags configured in this directory.
if (DIGITALCURLING_CLIENT_BUILD_TESTS)
    add_subdirectory(${PROJECT_SOURCE_DIR}/tests ${PROJECT_BINARY_DIR}/tests)
endif()
