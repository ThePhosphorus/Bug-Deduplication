cmake_minimum_required(VERSION 3.20)
project("Bug_Deduplication")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

# Find Python Libraries
set(Python3_FIND_VIRTUALENV ONLY)
find_package (Python3 COMPONENTS Interpreter Development)

if(NOT ${PYTHON3_FOUND} STREQUAL "TRUE")
	message("Could not find Python3")
	return()
endif()

# PyBind11
message(${Python3_RUNTIME_LIBRARY_DIRS})
set(pybind11_DIR
	/home/phos/projects/def-aloise/phos/bug_deduplication_stack_traces/ENV/lib/python3.9/site-packages/pybind11/share/cmake/pybind11)
find_package(pybind11 REQUIRED)

if (NOT ${pybind11_FOUND})
	message("Could not find Pybind11")
	return()
endif()

link_directories(${Python3_RUNTIME_LIBRARY_DIRS}/Library/lib)

find_package(TBB REQUIRED)

set (HEADERS_GLOBAL ${PROJECT_SOURCE_DIR}/dedup_mod_pybind/imports.hpp)
file(GLOB_RECURSE SRC_Methods ${PROJECT_SOURCE_DIR}/dedup_mod_pybind/method/*.cpp)
file(GLOB_RECURSE HEADERS_Methods ${PROJECT_SOURCE_DIR}/dedup_mod_pybind/method/*.hpp)
file(GLOB_RECURSE SRC_Util ${PROJECT_SOURCE_DIR}/dedup_mod_pybind/util/*.cpp)
file(GLOB_RECURSE HEADERS_Util ${PROJECT_SOURCE_DIR}/dedup_mod_pybind/util/*.hpp)
file(GLOB SRC_cmp_func ${PROJECT_SOURCE_DIR}/dedup_mod_pybind/*.cpp ${PROJECT_SOURCE_DIR}/dedup_mod_pybind/*.hpp)

set(METHOD_NAME "method")
set(${METHOD_NAME}_SRC
	${HEADERS_GLOBAL}
	${HEADERS_Methods}
	${SRC_Methods})
pybind11_add_module(${METHOD_NAME} MODULE ${${METHOD_NAME}_SRC})
target_link_libraries(${METHOD_NAME} PUBLIC pybind11::pybind11)

set(METHOD_LIB_NAME ${METHOD_NAME}_lib)
pybind11_add_module(${METHOD_LIB_NAME} STATIC ${${METHOD_NAME}_SRC})
target_link_libraries(${METHOD_LIB_NAME} PUBLIC pybind11::pybind11)


set(COMPARATOR_NAME "comparator")
set(${COMPARATOR_NAME}_SRC
	${HEADERS_GLOBAL}
	${HEADERS_Methods}
	${HEADERS_Util}
	${SRC_Util})
pybind11_add_module(${COMPARATOR_NAME} MODULE ${${COMPARATOR_NAME}_SRC})
target_link_libraries(${COMPARATOR_NAME} PUBLIC pybind11::pybind11)

set(COMPARATOR_LIB_NAME ${COMPARATOR_NAME}_lib)
pybind11_add_module(${COMPARATOR_LIB_NAME} STATIC ${${COMPARATOR_NAME}_SRC})
target_link_libraries(${COMPARATOR_LIB_NAME} PUBLIC pybind11::pybind11 )

set(CMP_FUNC_NAME "cmp_func")
set(${CMP_FUNC_NAME}_SRC
	${HEADERS_GLOBAL}
	${HEADERS_Methods}
	${SRC_Methods}
	${HEADERS_Util}
	${SRC_Util}
	${SRC_cmp_func})
pybind11_add_module(${CMP_FUNC_NAME} MODULE ${${CMP_FUNC_NAME}_SRC})
target_link_libraries(${CMP_FUNC_NAME}
	PUBLIC pybind11::pybind11 ${TBB_LIBRARIES})

# set(CMP_FUNC_LIB_NAME ${CMP_FUNC_NAME}_lib)
# pybind11_add_module(${CMP_FUNC_LIB_NAME} SHARED ${${CMP_FUNC_NAME}_SRC})
# target_link_libraries(${CMP_FUNC_LIB_NAME} PUBLIC pybind11::pybind11 ${TBB_LIBRARIES} ${METHOD_LIB_NAME} ${COMPARATOR_LIB_NAME})

install(TARGETS ${CMP_FUNC_NAME} DESTINATION .)
install(TARGETS ${METHOD_NAME} DESTINATION ./method)
install(TARGETS ${COMPARATOR_NAME} DESTINATION ./util)

# Tests
#include(BoostTestHelpers)
#set(TEST_DIR ${PROJECT_SOURCE_DIR}/tests/dedup_mod_pybind)
#
#find_boost_tests(TESTS "${TEST_DIR}/util" ${COMPARATOR_LIB_NAME})
#find_boost_tests(TESTS "${TEST_DIR}/method" ${METHOD_LIB_NAME})
#find_boost_tests(TESTS "${TEST_DIR}" ${CMP_FUNC_LIB_NAME})
#
#message("${TESTS}")
#
#install (TARGETS ${TESTS} DESTINATION ./tests)
