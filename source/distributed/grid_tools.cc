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
#include <deal.II/base/point.h>
#include <deal.II/grid/tria.h>
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

      //Degenerate cases: only one owned cell or we're at level 0
      if (lastc==firstc || firstc->level()*lastc->level()==0)
        {
          Point<spacedim> minp = firstc->center();
          Point<spacedim> maxp = firstc->center();
          for (unsigned int v=0; v<GeometryInfo<spacedim>::vertices_per_cell; ++v)
            {
              for ( unsigned int d=0; d<spacedim; ++d)
                {
                  minp[d] = std::min( minp[d], firstc->vertex(v)[d]);
                  maxp[d] = std::max( maxp[d], firstc->vertex(v)[d]);
                }
            }
          std::vector<BoundingBox<spacedim>> vBBox;
          vBBox.push_back(BoundingBox<spacedim>(std::make_pair( minp, maxp )));
          return vBBox;
        }

      //Now we look for parentc: the coarsest cell containing both firstc and lastc
      //This is the coarsest level at which we can separate the connected components
      Point<spacedim> lastc_center = lastc->center();
      typename parallel::distributed::Triangulation< dim, spacedim >::cell_iterator
      parentc = firstc;
      //tmpc shall contain the children of parentc from which firstc descends
      typename parallel::distributed::Triangulation< dim, spacedim >::cell_iterator
      tmpc = firstc;

      while (parentc->level()>0)
        {
          parentc = tmpc->parent();
          if (parentc->point_inside(lastc_center))
            break;
          else
            tmpc = parentc;
        }

      //Identifying the index of tmpc so that we can use firstc to
      //cycle over all locally_owned cells, starting from the first which is
      //locally owned

      unsigned int c;
      for (c=0; c<parentc->n_children(); ++c)
        {
          if (parentc->child(c)==tmpc)
            break;
        }

      // We now cycle over all children cells of parentc containing locally_owned
      // cells and, for each of these children cells, we build the bounding box
      // for the locally owned points inside it

      std::vector<BoundingBox<spacedim>> bounding_boxes;
      typename parallel::distributed::Triangulation< dim, spacedim >::cell_iterator
      stopc;

      for (; c<parentc->n_children(); ++c)
        {
          //First we need to find the "stopping point" i.e. the first
          //Active cell which is outside the current children cell

          if (c== parentc->n_children()-1)
            stopc = distributed_tria.last_active();
          else
            {
              stopc =  parentc->child(c+1);
              while (stopc->has_children())
                {
                  stopc = stopc->child(0);
                }
            }

          Point<spacedim> minp = firstc->center();
          Point<spacedim> maxp = firstc->center();

          bool flag = true; //Signal if we reach lastc

          for (; firstc<stopc && flag; firstc++)
            {
              for (unsigned int v=0; v<GeometryInfo<spacedim>::vertices_per_cell; ++v)
                {
                  for ( unsigned int d=0; d<spacedim; ++d)
                    {
                      minp[d] = std::min( minp[d], firstc->vertex(v)[d]);
                      maxp[d] = std::max( maxp[d], firstc->vertex(v)[d]);
                    }
                }
              if (firstc==lastc)
                flag = false;
            }

          if (minp != maxp)
            bounding_boxes.push_back( BoundingBox<spacedim>(std::make_pair(minp,maxp)) );

          if (!flag)
            break; //If flag is false we've reached lastc
        }
      return bounding_boxes;
    }
  }
}

#include "grid_tools.inst"

DEAL_II_NAMESPACE_CLOSE

#endif
