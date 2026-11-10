# CTest: src/build_id.cmake removes the "g" before the commit hash and keeps -dirty.
# Run with -DSCRIPT=<src/build_id.cmake> -DWORK_DIR=<dir>.
function(check input expected)
    set(out "${WORK_DIR}/build_id_test.hpp")
    execute_process(COMMAND ${CMAKE_COMMAND} -DSOURCE_DIR=${WORK_DIR} -DOUTPUT=${out}
                            -DBUILD_ID_OVERRIDE=${input} -P ${SCRIPT}
                    RESULT_VARIABLE rc)
    if (NOT rc EQUAL 0)
        message(FATAL_ERROR "build_id.cmake failed for ${input}")
    endif()
    file(READ "${out}" content)
    if (NOT content MATCHES "BUILD_ID \"${expected}\"")
        message(FATAL_ERROR "${input}: expected ${expected}, got ${content}")
    endif()
endfunction()

check("v2026.07-67-ga604b03" "v2026.07-67-a604b03")
check("v2026.07-67-ga604b03-dirty" "v2026.07-67-a604b03-dirty")
check("v26.11.0" "v26.11.0")
check("v26.11.0-dirty" "v26.11.0-dirty")
check("a604b03" "a604b03")
