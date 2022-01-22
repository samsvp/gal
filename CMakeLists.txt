find_package(ArrayFire)
add_executable(main main.cpp)

# To use Unified backend, do the following.
# Unified backend lets you choose the backend at runtime
target_link_libraries(main ArrayFire::afopencl)

# copy scripts folder into build
add_custom_command(TARGET main POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_SOURCE_DIR}/scripts/ $<TARGET_FILE_DIR:main>/scripts)