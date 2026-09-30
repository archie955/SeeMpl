# Fetch Unity testing framework
include(FetchContent)

FetchContent_Declare(
    unity
    GIT_REPOSITORY https://github.com/ThrowTheSwitch/Unity.git
    GIT_TAG        v2.6.0
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(unity)

# test 1d value
# add_executable(test_1dValue test_1dValue.c)
# target_link_libraries(test_matrix PRIVATE
#     SeeMpl::SeeMpl
#     unity::framework
# )
# add_test(NAME test_matrix COMMAND test_matrix)
# 
# # setup test network
# add_executable(test_network test_network.c)
# target_link_libraries(test_network PRIVATE
#     SeeMpl::SeeMpl
#     unity::framework
# )
# add_test(NAME test_network COMMAND test_network)