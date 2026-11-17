# Set your client name
set(DIGITALCURLING_CLIENT_NAME "CNNClient")
# Set your client version (format: major.minor.patch). Calendar version YY.MM.N,
# changed only at a release; see "Versions" in README.md.
set(DIGITALCURLING_CLIENT_VERSION "26.11.0")

# set additional source files for the client (optional)
set(DIGITALCURLING_CLIENT_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/engine/cnn_engine.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/engine/apply_move.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/engine/model_identity.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/engine/skip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/engine/uctnode.cpp"
)
# set your client to use the plugin loader (optional, default: OFF)
set(DIGITALCURLING_CLIENT_USE_LOADER ON)
