file(READ "${INPUT}" PTX HEX)
# Numeric bytes avoid MSVC's string literal size limit and preserve PTX exactly.
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," PTX "${PTX}")
string(REPEAT "0x[0-9a-f][0-9a-f]," 16 ROW)
string(REGEX REPLACE "(${ROW})" "\\1\n" PTX "${PTX}")
file(WRITE "${OUTPUT}" "// Generated OptiX program bytes.\nstatic constexpr unsigned char kRtrtOptixPtx[] = {\n${PTX}0};\n")
