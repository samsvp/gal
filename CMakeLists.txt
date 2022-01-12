find_package(ArrayFire)
add_executable(main main.cpp)

# To use Unified backend, do the following.
# Unified backend lets you choose the backend at runtime
target_link_libraries(main ArrayFire::afopencl)