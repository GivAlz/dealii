/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2018 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------
 */

// Check that we can construct boost R-trees using containers and iterator
// ranges, and perform a trivial query on them.

#include <deal.II/base/patterns.h>
#include <deal.II/base/point.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/numerics/rtree.h>

#include "../tests.h"

namespace bgi = boost::geometry::index;

int
main()
{
  initlog();
  const unsigned int    N = 30;
  std::vector<Point<2>> points(N);
  for (auto &p : points)
    p = random_point<2>();

  auto tree  = pack_rtree(points.begin(), points.end());
  auto tree2 = pack_rtree(points);

  Point<2>   p0(0, 0);
  Point<2>   p1(.4, .7111);
  Segment<2> segment(p0, p1);

  {
    decltype(points) nearest;
    tree.query(bgi::nearest(segment, 3), std::back_inserter(nearest));

    if (nearest.size() != 3)
      deallog << "Not OK." << std::endl;
    else
      deallog << "OK." << std::endl;
  }
  {
    decltype(points) nearest;
    tree2.query(bgi::nearest(segment, 3), std::back_inserter(nearest));

    if (nearest.size() != 3)
      deallog << "Not OK." << std::endl;
    else
      deallog << "OK." << std::endl;
  }
}