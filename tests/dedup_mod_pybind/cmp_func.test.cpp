#define BOOST_TEST_MODULE CMP_FUNC
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/test/unit_test.hpp>

#include <vector>

#include "../../dedup_mod_pybind/cmp_func.hpp"

boost::random::mt19937 seed;

BOOST_AUTO_TEST_CASE(TestTrue) { BOOST_CHECK(true); }
// There aren't actual tests yet
