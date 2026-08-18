// Compiles doctest's implementation and provides main().
//
// Kept in its own translation unit so the (large) doctest runtime is built
// once rather than once per test file. Every other test file includes
// third_party/doctest.h without defining the IMPLEMENT macro.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "third_party/doctest.h"
