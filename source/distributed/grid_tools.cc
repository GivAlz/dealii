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



#include <deal.II/base/bounding_box.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/point.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_tools.h>

#ifdef DEAL_II_WITH_P4EST

#include <vector>
#include <cmath>

DEAL_II_NAMESPACE_OPEN

namespace parallel
{
  namespace GridTools
  {
    template < int dim, int spacedim>
    std::vector<BoundingBox<spacedim> >
    compute_locally_owned_bounding_box
    (const parallel::distributed::Triangulation< dim, spacedim > &distributed_tria)
    {
      // In order to return the boundary points of the locally_owned cells correctly
      // we need to handle the possibility of a second connected component.
      //
      // Assuming the triangulation is distributed following the morton-z curve order
      // we know that, if there are two connected components, then the first and the
      // last locally owned cell belong to different components.
      //
      // Thus we begin by looking for those cells which we call firstc and lastc

      typename parallel::distributed::Triangulation< dim, spacedim >::active_cell_iterator
      firstc = distributed_tria.begin_active();
      typename parallel::distributed::Triangulation< dim, spacedim >::active_cell_iterator
      lastc = distributed_tria.last_active();

      while ( ! firstc->is_locally_owned() )
        ++firstc;


      while ( ! lastc->is_locally_owned() )
        --lastc;

#ifdef DEBUG
      typename parallel::distributed::Triangulation< dim, spacedim >::active_cell_iterator
      ic = distributed_tria.begin_active();
      typename parallel::distributed::Triangulation< dim, spacedim >::active_cell_iterator
      endc = distributed_tria.last_active();
      for(;ic<endc;++ic)
          if(ic<firstc || ic>lastc)
          Assert( ! ic->is_locally_owned(),
                 ExcMessage ( "Error: locally owned cells outside the studied interval") );
#endif
      //Now we look for parentc: the coarsest cell containing both firstc and lastc
      //This is the coarsest level at which we can separate the connected components
      typename parallel::distributed::Triangulation< dim, spacedim >::cell_iterator
      parent_firstc = firstc;
      //tmpc shall contain the children of parentc from which firstc descends
      typename parallel::distributed::Triangulation< dim, spacedim >::cell_iterator
      parent_lastc = lastc;

      //First we reach the coarsest common level
      if(parent_firstc->level() > parent_lastc->level() )
          while( parent_lastc->level() < parent_firstc->level() )
              parent_lastc = parent_lastc->parent();
      else if(parent_firstc->level() < parent_lastc->level())
          while( parent_lastc->level() > parent_firstc->level() )
              parent_firstc = parent_firstc->parent();

      //Now we look for the coarsest cell containing both firstc and lastc
      //This is the coarsest level at which we can separate the connected components

      while(parent_firstc != parent_lastc && parent_firstc->level()>0)
      {
          parent_firstc = parent_firstc->parent();
          parent_lastc = parent_lastc->parent();
      }


      std::vector< BoundingBox < spacedim > > bounding_boxes;
      if(parent_firstc == parent_lastc)
      {
          //If the two are the same we now create a bounding box for each child:
          for(unsigned int c=0; c<parent_firstc->n_children(); ++c)
          {
              std::vector < typename parallel::distributed::Triangulation< dim, spacedim >::active_cell_iterator >
                      local_cells =
                      dealii::GridTools::get_active_child_cells < typename parallel::distributed::Triangulation< dim, spacedim > >
                      (parent_firstc->child(c));
              bool no_bbox = true;
              Point<spacedim> minp;
              Point<spacedim> maxp;
              for(unsigned int i=0; i< local_cells.size();++i)
              {
                  if(local_cells[i]->is_locally_owned())
                  {
                      if(no_bbox)
                      {
                          minp = local_cells[i]->center();
                          maxp = local_cells[i]->center();
                          no_bbox = false;
                      }
                      for (unsigned int v=0; v<GeometryInfo<spacedim>::vertices_per_cell; ++v)
                          for ( unsigned int d=0; d<spacedim; ++d)
                            {
                              minp[d] = std::min( minp[d], local_cells[i]->vertex(v)[d]);
                              maxp[d] = std::max( maxp[d], local_cells[i]->vertex(v)[d]);
                            }
                  }
              }
              if( ! no_bbox)
                  bounding_boxes.push_back( BoundingBox < spacedim > (std::make_pair(minp,maxp)) );
          }
      }
      else
      {
          for(auto cell: distributed_tria.cell_iterators_on_level(0))
          {
              std::vector < typename parallel::distributed::Triangulation< dim, spacedim >::active_cell_iterator >
                      local_cells =
                      dealii::GridTools::get_active_child_cells < typename parallel::distributed::Triangulation< dim, spacedim > >
                      (cell);
              bool no_bbox = true;
              Point<spacedim> minp;
              Point<spacedim> maxp;
              for(unsigned int i=0; i< local_cells.size();++i)
              {
                  if(local_cells[i]->is_locally_owned())
                  {
                      if(no_bbox)
                      {
                          minp = local_cells[i]->center();
                          maxp = local_cells[i]->center();
                          no_bbox = false;
                      }
                      for (unsigned int v=0; v<GeometryInfo<spacedim>::vertices_per_cell; ++v)
                          for ( unsigned int d=0; d<spacedim; ++d)
                            {
                              minp[d] = std::min( minp[d], local_cells[i]->vertex(v)[d]);
                              maxp[d] = std::max( maxp[d], local_cells[i]->vertex(v)[d]);
                            }
                  }
              }
              if( ! no_bbox)
                  bounding_boxes.push_back( BoundingBox < spacedim > (std::make_pair(minp,maxp)) );
          }
      }
      return bounding_boxes;
    }
  }
}

#include "grid_tools.inst"

DEAL_II_NAMESPACE_CLOSE

#endif
