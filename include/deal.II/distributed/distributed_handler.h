// ---------------------------------------------------------------------
//
// Copyright (C) 2018 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------

#ifndef dealii_distributed_handler_h
#define dealii_distributed_handler_h

#include <deal.II/base/bounding_box.h>
#include <deal.II/base/config.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/point.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/tria.h>

DEAL_II_NAMESPACE_OPEN

template <int dim, int spacedim>
class DistributedHandler
{
public:
  /**
   * Constructor: as input it needs a parallel distributed triangulation
   */
  DistributedHandler(const parallel::distributed::Triangulation<dim,spacedim> &triangulation);

  /**
   * Manually update the local bounding box vector
   */
  void update_local_description(std::vector<BoundingBox<spacedim>> local_description);

  /**
   * Exchange global description
   */
  void update_global_description();

private:
  // Triangulation which is handled by the object
  const parallel::distributed::Triangulation<dim,spacedim> &triangulation;

  // Mpi Communicator
  MPI_Comm mpi_communicator;

  // Local description of the mesh using bounding boxes
  // In the future a tree should be used
  std::vector<BoundingBox<spacedim>> local_description;

  // Global description of the mesh using bounding boxes.
  // In the future a tree should be used
  std::vector< std::vector<BoundingBox<spacedim>>> global_description;
};
/*------------------------ Inline functions: DistributedHandler --------------------*/

#ifndef DOXYGEN


template <int dim, int spacedim>
    inline DistributedHandler<dim,spacedim>::DistributedHandler(
           const parallel::distributed::Triangulation<dim,spacedim> &triangulation)
{
  this->triangulation = triangulation;
  this->mpi_communicator = triangulation->get_mpi_communicator();
}



template <int dim, int spacedim>
    inline DistributedHandler<dim,spacedim>::update_local_description
        (std::vector<BoundingBox<spacedim>> local_description)
{
  this->local_description = local_description;
}



template <int dim, int spacedim>
    inline DistributedHandler<dim,spacedim>::update_global_description()
{
    this->global_bboxes = Utilities::MPI::all_gather(this->mpi_communicator,
                                                     this->local_description);
}


#endif // DOXYGEN
DEAL_II_NAMESPACE_CLOSE




#endif
