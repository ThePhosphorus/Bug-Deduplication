# Inspired from https://eb2.co/blog/2015/06/driving-boost.test-with-cmake/#:~:text=To%20begin%2C%20Boost.Test%20must%20be%20included%20in%20the,by%20CTest%20from%20a%20single%20test%20source%20file.
set(Boost_USE_STATIC_LIBS ON)
find_package(Boost COMPONENTS unit_test_framework REQUIRED)

macro(add_boost_test TESTS_VAR SOURCE_FILE_NAME DEPENDENCY_LIB)
    get_filename_component(TEST_EXECUTABLE_NAME_STUB ${SOURCE_FILE_NAME} NAME_WE)
    set(TEST_EXECUTABLE_NAME test_${TEST_EXECUTABLE_NAME_STUB})

    add_executable(${TEST_EXECUTABLE_NAME} ${SOURCE_FILE_NAME})
    target_include_directories(${TEST_EXECUTABLE_NAME} PUBLIC ${Boost_INCLUDE_DIRS})
    target_link_libraries(${TEST_EXECUTABLE_NAME} PUBLIC ${DEPENDENCY_LIB} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} pybind11::embed)
    list(APPEND ${TESTS_VAR} ${TEST_EXECUTABLE_NAME})
endmacro()

macro(find_boost_tests TESTS_VAR TEST_DIR_NAME DEPENDENCY_LIB)
    file(GLOB TEST_FILES ${TEST_DIR_NAME}/*.test.cpp)
        foreach(TEST_FILE ${TEST_FILES})
            add_boost_test(${TESTS_VAR} ${TEST_FILE} ${DEPENDENCY_LIB})
        endforeach()
endmacro()
