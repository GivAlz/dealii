// ---------------------------------------------------------------------
//
// Copyright (C) 2017 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the deal.II distribution.
//
// ---------------------------------------------------------------------


// test for the function compute_locally_owned_bounding_box : on various domains and
// mpi configurations to vary the shape of the various domains


#include "../tests.h"
#include <deal.II/base/logstream.h>
#include <deal.II/base/bounding_box.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>
#include <deal.II/base/utilities.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/distributed/grid_tools.h>

template <int dim, int spacedim=dim >
void test_hypercube()
{
  const MPI_Comm &mpi_communicator = MPI_COMM_WORLD;
  deallog << "spacedim = " << spacedim << std::endl;

  parallel::distributed::Triangulation<spacedim> tria(mpi_communicator);
  GridGenerator::hyper_cube (tria);
  tria.refine_global(3);

  std::vector<BoundingBox<spacedim>> local_bbox =
                                    parallel::GridTools::compute_locally_owned_bounding_box<spacedim>(tria);

  deallog << "Computed bounding boxes:" << std::endl;

  for (unsigned int i=0; i< local_bbox.size(); ++i)
    deallog << local_bbox[i].get_boundary_points().first <<
            " " << local_bbox[i].get_boundary_points().second << std::endl;

  //Checking if all the points are inside the bounding boxes
  bool check = true;

  typename parallel::distributed::Triangulation< dim, spacedim >::active_cell_iterator
  cell = tria.begin_active();
  typename parallel::distributed::Triangulation< dim, spacedim >::active_cell_iterator
  endc = tria.last_active();

  //Looking if every point is at least inside a bounding box
  for (; cell<endc; ++cell)
    if (cell->is_locally_owned())
      for (unsigned int v=0; v<GeometryInfo<spacedim>::vertices_per_cell; ++v)
        {
          bool local = false;
          for (unsigned int i=0; i< local_bbox.size(); ++i)
            {
              if (local_bbox[i].point_inside(cell->vertex(v)))
                local = true;
            }
          if (! local)
            {
              check = false;
              deallog << "Point outside " << cell->vertex(v) << std::endl;
              break;
            }
        }

  deallog << "Bounding Boxes surround locally_owned cells: " << check << std::endl;
}

int main (int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv, 1);
  MPILogInitAll log;

  test_hypercube<2> ();
  test_hypercube<3> ();

  return 0;
}
