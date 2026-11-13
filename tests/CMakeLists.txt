# Unit tests (enabled with -DDIGITALCURLING_CLIENT_BUILD_TESTS=ON).
# Tests run on CPU only and need no network, GPU or model files.

include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    # renovate: datasource=github-tags depName=google/googletest
    GIT_TAG        v1.17.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

# One executable per unit so that units with header-defined (non-inline)
# functions such as readcsv.hpp don't collide at link time.
function(digitalcurling_client_add_test name)
    add_executable(${name} ${CMAKE_CURRENT_SOURCE_DIR}/${name}.cpp)
    target_link_libraries(${name} PRIVATE ${ARGN} GTest::gtest_main)
    target_compile_features(${name} PRIVATE cxx_std_17)
    gtest_discover_tests(${name}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        DISCOVERY_TIMEOUT 60
    )
endfunction()

digitalcurling_client_add_test(test_readcsv)
target_include_directories(test_readcsv PRIVATE ${PROJECT_SOURCE_DIR}/src)

digitalcurling_client_add_test(test_utility ${DIGITALCURLING_CLIENT_ENGINE_LIB})
digitalcurling_client_add_test(test_apply_move ${DIGITALCURLING_CLIENT_ENGINE_LIB})
digitalcurling_client_add_test(test_uctnode ${DIGITALCURLING_CLIENT_ENGINE_LIB})
digitalcurling_client_add_test(test_search_policy ${DIGITALCURLING_CLIENT_ENGINE_LIB})
digitalcurling_client_add_test(test_client_helpers ${DIGITALCURLING_CLIENT_ENGINE_LIB})

digitalcurling_client_add_test(test_model_identity ${DIGITALCURLING_CLIENT_ENGINE_LIB})
target_include_directories(test_model_identity PRIVATE ${PROJECT_SOURCE_DIR}/src)

add_test(NAME build_id_strips_g
    COMMAND ${CMAKE_COMMAND} -DSCRIPT=${PROJECT_SOURCE_DIR}/src/build_id.cmake
            -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/test_build_id.cmake)
