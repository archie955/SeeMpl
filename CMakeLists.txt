cmake_minimum_required(VERSION 3.21)

project(SeeMpl
    VERSION 0.0.1
    DESCRIPTION "A simple MLP library written in C"
    LANGUAGES C
)

# recommended standards
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# export compile_commands.json for clang-tidy
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# set build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# sanitizer
if(ENABLE_SANITIZE)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
endif()

# output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# options so I don't accidentally do everything at once
option(BUILD_TESTS    "Build unit tests"         OFF)
option(ENABLE_TIDY    "Enable clang-tidy checks" OFF)
option(ENABLE_SANITIZE "Enable ASan + UBSan"     OFF)
option(BUILD_PYTHON "Build python bindings" OFF)

# packaging
add_library(SeeMpl STATIC
    src/1dValue.c
    src/Layer.c
    src/MLP.c
    src/Neuron.c
    src/Tape.c
)
set_target_properties(SeeMpl PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(SeeMpl::SeeMpl ALIAS SeeMpl)

target_include_directories(SeeMpl
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(SeeMpl PUBLIC m)

add_executable(seempl_demo src/main.c)
target_link_libraries(seempl_demo PRIVATE SeeMpl)

# Python binding
if (BUILD_PYTHON)
    find_package(Python3 REQUIRED COMPONENTS Development.Module)
    add_library(cnn MODULE src/python_binding.c)
    target_link_libraries(cnn PRIVATE SeeMpl Python3::Module)
    set_target_properties(cnn PROPERTIES PREFIX "")
endif()

# clang tidy
if(ENABLE_TIDY)
    find_program(CLANG_TIDY_BIN clang-tidy REQUIRED)
    set_target_properties(SeeMpl PROPERTIES
        C_CLANG_TIDY "${CLANG_TIDY_BIN}"
    )
endif()

# testing
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# export
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS SeeMpl
    EXPORT SeeMplTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/SeeMpl
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/SeeMpl
)

install(EXPORT SeeMplTargets
    FILE SeeMplTargets.cmake
    NAMESPACE SeeMpl::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SeeMpl
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/SeeMplConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/SeeMplConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SeeMpl
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/SeeMplConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/SeeMplConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/SeeMplConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SeeMpl
)