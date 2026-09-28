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

# output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# following recommendation
option(BUILD_TESTS    "Build unit tests"         ON)
option(ENABLE_TIDY    "Enable clang-tidy checks" OFF)
option(ENABLE_SANITIZE "Enable ASan + UBSan"     OFF)

# packaging
add_library(SeeMpl STATIC
    src/main.c
)
add_library(SeeMpl::SeeMpl ALIAS SeeMpl)

target_include_directories(SeeMpl
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# sanitizer
if(ENABLE_SANITIZE)
    target_compile_options(SeeMpl PRIVATE -fsanitize=address,undefined)
    target_link_options(SeeMpl    PRIVATE -fsanitize=address,undefined)
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