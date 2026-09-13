file(READ "${INPUT}" PTX)
file(WRITE "${OUTPUT}" "// Generated from the shared RTRT shading implementation.\nstatic constexpr char kRtrtOptixPtx[] = R\"RTRT_PTX(${PTX})RTRT_PTX\";\n")
