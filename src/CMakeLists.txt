function(target_compile_definition_if_exists target_name definition variable_name)
    set(variable_value "${${variable_name}}")
    if(NOT variable_value STREQUAL "")
        target_compile_definitions(${target_name} PRIVATE ${definition}=${variable_value})
    endif()
endfunction()

# Torch (libtorch)
set(LIBTORCH_DIR /usr/local/libtorch/)
set(CAFFE2_USE_CUDNN 1)
set(CAFFE2_USE_CUSPARSELT 1)
list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_DIR})
find_package(Torch REQUIRED)

# OpenMP
find_package(OpenMP)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# --- Build client ---
# sources
add_executable(${PROJECT_NAME}
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/client_setup.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/client/client_base.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/client/client_factory.cpp
    ${DIGITALCURLING_CLIENT_SOURCES}
)
target_include_directories(${PROJECT_NAME}
    PRIVATE ${CMAKE_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(${PROJECT_NAME} PRIVATE CLI11::CLI11 httplib ${TORCH_LIBRARIES})
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
