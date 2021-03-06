// ---------------------------------------------------------------------
//
// Copyright (C) 2016 - 2018 by the deal.II authors
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

#include <cell_accessor_wrapper.h>

#include <boost/python.hpp>
#include <point_wrapper.h>
#include <triangulation_wrapper.h>

DEAL_II_NAMESPACE_OPEN

namespace python
{
  namespace internal
  {
    template <int dim, int spacedim>
    void set_refine_flag(const std::string &refinement_case,
                         void              *cell_accessor)
    {
      CellAccessor<dim,spacedim> *cell =
        static_cast<CellAccessor<dim,spacedim>*>(cell_accessor);

      std::unique_ptr<RefinementCase<dim>> ref_case;
      if (refinement_case.compare("isotropic") == 0)
        ref_case.reset(new RefinementCase<dim>(
                         RefinementPossibilities<dim>::Possibilities::isotropic_refinement));
      else if (refinement_case.compare("no_refinement") == 0)
        ref_case.reset(new RefinementCase<dim>(
                         RefinementPossibilities<dim>::Possibilities::no_refinement));
      else if (refinement_case.compare("cut_x") == 0)
        ref_case.reset(new RefinementCase<dim>(
                         RefinementPossibilities<dim>::Possibilities::cut_x));
      else if (refinement_case.compare("cut_y") == 0)
        ref_case.reset(new RefinementCase<dim>(
                         RefinementPossibilities<dim>::Possibilities::cut_y));
      else if (refinement_case.compare("cut_xy") == 0)
        ref_case.reset(new RefinementCase<dim>(
                         RefinementPossibilities<dim>::Possibilities::cut_xy));
#if dim==3
      else if (refinement_case.compare("cut_z") == 0)
        ref_case.reset(new RefinementCase<3>(
                         RefinementPossibilities<3>::Possibilities::cut_z));
      else if (refinement_case.compare("cut_xz") == 0)
        ref_case.reset(new RefinementCase<3>(
                         RefinementPossibilities<3>::Possibilities::cut_xz));
      else if (refinement_case.compare("cut_yz") == 0)
        ref_case.reset(new RefinementCase<3>(
                         RefinementPossibilities<3>::Possibilities::cut_yz));
      else if (refinement_case.compare("cut_xyz") == 0)
        ref_case.reset(new RefinementCase<3>(
                         RefinementPossibilities<3>::Possibilities::cut_xyz));
#endif
      else
        AssertThrow(false,
                    ExcMessage("Unknown refinement possibility."));

      cell->set_refine_flag(*ref_case);
    }



    template <int dim, int spacedim>
    std::string get_refine_flag(const void *cell_accessor)
    {
      const CellAccessor<dim,spacedim> *cell =
        static_cast<const CellAccessor<dim,spacedim>*>(cell_accessor);

      std::string refine_flag;
      RefinementCase<dim> ref_case = cell->refine_flag_set();
      switch (static_cast<int>(ref_case))
        {
        case (0) :
        {
          refine_flag = "no_refinement";
          break;
        }
        case (1) :
        {
          refine_flag = "cut_x";
          break;
        }
        case (2) :
        {
          refine_flag = "cut_y";
          break;
        }
        case (3) :
        {
          refine_flag = "cut_xy";
          break;
        }
        case (4) :
        {
          refine_flag = "cut_z";
          break;
        }
        case (5) :
        {
          refine_flag = "cut_xz";
          break;
        }
        case (6) :
        {
          refine_flag = "cut_yz";
          break;
        }
        case (7) :
        {
          refine_flag = "cut_xyz";
          break;
        }
        default :
        {
          AssertThrow(false, ExcMessage("Internal error."));
        }
        }

      return refine_flag;
    }



    template <int dim, int spacedim>
    void set_coarsen_flag(const bool coarsen_flag, void *cell_accessor)
    {
      CellAccessor<dim,spacedim> *cell =
        static_cast<CellAccessor<dim,spacedim>*>(cell_accessor);
      if (coarsen_flag == true)
        cell->set_coarsen_flag();
      else
        cell->clear_coarsen_flag();
    }



    template <int dim, int spacedim>
    bool get_coarsen_flag(const void *cell_accessor)
    {
      const CellAccessor<dim,spacedim> *cell =
        static_cast<const CellAccessor<dim,spacedim>*>(cell_accessor);

      return cell->coarsen_flag_set();
    }



    template <int dim, int spacedim>
    PointWrapper get_barycenter(const void *cell_accessor)
    {
      const CellAccessor<dim,spacedim> *cell =
        static_cast<const CellAccessor<dim,spacedim>*>(cell_accessor);
      Point<spacedim> barycenter = cell->barycenter();
      boost::python::list barycenter_list;
      for (int i=0; i<dim; ++i)
        barycenter_list.append(barycenter[i]);

      return PointWrapper(barycenter_list);
    }



    template <int dim, int spacedim>
    void set_material_id(const int material_id, void *cell_accessor)
    {
      CellAccessor<dim,spacedim> *cell =
        static_cast<CellAccessor<dim,spacedim>*>(cell_accessor);
      cell->set_material_id(material_id);
    }



    template <int dim, int spacedim>
    int get_material_id(const void *cell_accessor)
    {
      const CellAccessor<dim,spacedim> *cell =
        static_cast<const CellAccessor<dim,spacedim>*>(cell_accessor);

      return cell->material_id();
    }



    template <int dim, int spacedim>
    void set_vertex(const int     i,
                    PointWrapper &point_wrapper,
                    void         *cell_accessor)
    {
      CellAccessor<dim,spacedim> *cell =
        static_cast<CellAccessor<dim,spacedim>*>(cell_accessor);
      Point<spacedim> *point =
        static_cast<Point<spacedim>*>(point_wrapper.get_point());

      cell->vertex(i) = *point;
    }



    template <int dim, int spacedim>
    PointWrapper get_vertex(const int i, const void *cell_accessor)
    {
      const CellAccessor<dim,spacedim> *cell =
        static_cast<const CellAccessor<dim,spacedim>*>(cell_accessor);
      Point<spacedim> vertex = cell->vertex(i);

      boost::python::list coordinates;
      for (int i=0; i<spacedim; ++i)
        coordinates.append(vertex[i]);

      return PointWrapper(coordinates);
    }



    template <int dim, int spacedim>
    void set_manifold_id(const int manifold_id, void *cell_accessor)
    {
      CellAccessor<dim,spacedim> *cell =
        static_cast<CellAccessor<dim,spacedim>*>(cell_accessor);
      cell->set_manifold_id(manifold_id);
    }



    template <int dim, int spacedim>
    int get_manifold_id(const void *cell_accessor)
    {
      const CellAccessor<dim,spacedim> *cell =
        static_cast<const CellAccessor<dim,spacedim>*>(cell_accessor);
      return cell->manifold_id();
    }
  }



  CellAccessorWrapper::CellAccessorWrapper(const CellAccessorWrapper &other)
    :
    dim(other.dim),
    spacedim(other.spacedim)
  {
    if ((dim == 2) && (spacedim == 2))
      {
        CellAccessor<2,2> *other_cell =
          static_cast<CellAccessor<2,2>*>(other.cell_accessor);
        cell_accessor = new CellAccessor<2,2>(*other_cell);
      }
    else if ((dim == 2) && (spacedim == 3))
      {
        CellAccessor<2,3> *other_cell =
          static_cast<CellAccessor<2,3>*>(other.cell_accessor);
        cell_accessor = new CellAccessor<2,3>(*other_cell);
      }
    else if ((dim == 3) && (spacedim == 3))
      {
        CellAccessor<3,3> *other_cell =
          static_cast<CellAccessor<3,3>*>(other.cell_accessor);
        cell_accessor = new CellAccessor<3,3>(*other_cell);
      }
    else
      AssertThrow(false, ExcMessage("Wrong dim-spacedim combination."));
  }




  CellAccessorWrapper::CellAccessorWrapper(TriangulationWrapper &triangulation_wrapper,
                                           const int             level,
                                           const int             index)
  {
    dim = triangulation_wrapper.get_dim();
    spacedim = triangulation_wrapper.get_spacedim();
    if ((dim == 2) && (spacedim == 2))
      {
        Triangulation<2,2> *tmp = static_cast<Triangulation<2,2>*> (
                                    triangulation_wrapper.get_triangulation());
        cell_accessor = new CellAccessor<2,2>(tmp ,level, index);
      }
    else if ((dim == 2) && (spacedim == 3))
      {
        Triangulation<2,3> *tmp = static_cast<Triangulation<2,3>*> (
                                    triangulation_wrapper.get_triangulation());
        cell_accessor = new CellAccessor<2,3>(tmp, level, index);
      }
    else if ((dim == 3) && (spacedim == 3))
      {
        Triangulation<3,3> *tmp = static_cast<Triangulation<3,3>*> (
                                    triangulation_wrapper.get_triangulation());
        cell_accessor = new CellAccessor<3,3>(tmp, level, index);
      }
    else
      AssertThrow(false, ExcMessage("Wrong dim-spacedim combination."));
  }



  CellAccessorWrapper::~CellAccessorWrapper()
  {
    if (dim != -1)
      {
        if ((dim == 2) && (spacedim == 2))
          {
            // We cannot call delete on a void pointer so cast the void pointer back
            // first.
            CellAccessor<2,2> *tmp =
              static_cast<CellAccessor<2,2>*>(cell_accessor);
            delete tmp;
          }
        else if ((dim == 2) && (spacedim == 3))
          {
            CellAccessor<3,3> *tmp =
              static_cast<CellAccessor<3,3>*>(cell_accessor);
            delete tmp;
          }
        else
          {
            CellAccessor<3,3> *tmp =
              static_cast<CellAccessor<3,3>*>(cell_accessor);
            delete tmp;
          }

        dim = -1;
        spacedim = -1;
        cell_accessor = nullptr;
      }
  }



  void CellAccessorWrapper::set_refine_flag(const std::string &refinement_case)
  {
    if ((dim == 2) && (spacedim == 2))
      internal::set_refine_flag<2,2>(refinement_case, cell_accessor);
    else if ((dim == 2) && (spacedim == 3))
      internal::set_refine_flag<2,3>(refinement_case, cell_accessor);
    else
      internal::set_refine_flag<3,3>(refinement_case, cell_accessor);
  }



  std::string CellAccessorWrapper::get_refine_flag() const
  {
    if ((dim == 2) && (spacedim == 2))
      return internal::get_refine_flag<2,2>(cell_accessor);
    else if ((dim == 2) && (spacedim == 3))
      return internal::get_refine_flag<2,3>(cell_accessor);
    else
      return internal::get_refine_flag<3,3>(cell_accessor);
  }



  void CellAccessorWrapper::set_coarsen_flag(const bool coarsen_flag)
  {
    if ((dim == 2) && (spacedim == 2))
      internal::set_coarsen_flag<2,2>(coarsen_flag, cell_accessor);
    else if ((dim == 2) && (spacedim == 3))
      internal::set_coarsen_flag<2,3>(coarsen_flag, cell_accessor);
    else
      internal::set_coarsen_flag<3,3>(coarsen_flag, cell_accessor);
  }



  bool CellAccessorWrapper::get_coarsen_flag() const
  {
    if ((dim == 2) && (spacedim == 2))
      return internal::get_coarsen_flag<2,2>(cell_accessor);
    else if ((dim == 2) && (spacedim == 3))
      return internal::get_coarsen_flag<2,3>(cell_accessor);
    else
      return internal::get_coarsen_flag<3,3>(cell_accessor);
  }


  PointWrapper CellAccessorWrapper::get_barycenter() const
  {
    if ((dim == 2) && (spacedim == 2))
      return internal::get_barycenter<2,2>(cell_accessor);
    else if ((dim == 2) && (spacedim == 3))
      return internal::get_barycenter<2,3>(cell_accessor);
    else
      return internal::get_barycenter<3,3>(cell_accessor);
  }



  void CellAccessorWrapper::set_material_id(const int material_id)
  {
    AssertThrow(static_cast<types::material_id>(material_id) < numbers::invalid_material_id,
                ExcMessage("material_id is too large."));
    if ((dim == 2) && (spacedim == 2))
      return internal::set_material_id<2,2>(material_id, cell_accessor);
    else if ((dim == 2) && (spacedim == 3))
      return internal::set_material_id<2,3>(material_id, cell_accessor);
    else
      return internal::set_material_id<3,3>(material_id, cell_accessor);
  }



  int CellAccessorWrapper::get_material_id() const
  {
    if ((dim == 2) && (spacedim == 2))
      return internal::get_material_id<2,2>(cell_accessor);
    else if ((dim == 2) && (spacedim == 3))
      return internal::get_material_id<2,3>(cell_accessor);
    else
      return internal::get_material_id<3,3>(cell_accessor);
  }



  void CellAccessorWrapper::set_vertex(const int     i,
                                       PointWrapper &point_wrapper)
  {
    AssertThrow(i<std::pow(2,dim),
                ExcVertexDoesNotExist(i, std::pow(2,dim)));
    if ((dim == 2) && (spacedim == 2))
      internal::set_vertex<2,2>(i, point_wrapper, cell_accessor);
    else if ((dim == 2) && (spacedim == 3))
      internal::set_vertex<2,3>(i, point_wrapper, cell_accessor);
    else
      internal::set_vertex<3,3>(i, point_wrapper, cell_accessor);
  }



  PointWrapper CellAccessorWrapper::get_vertex(const int i) const
  {
    AssertThrow(i<std::pow(2,dim),
                ExcVertexDoesNotExist(i, std::pow(2,dim)));
    if ((dim == 2) && (spacedim == 2))
      return internal::get_vertex<2,2>(i, cell_accessor);
    else if ((dim == 2) && (spacedim == 3))
      return internal::get_vertex<2,3>(i, cell_accessor);
    else
      return internal::get_vertex<3,3>(i, cell_accessor);
  }



  void CellAccessorWrapper::set_manifold_id(const int manifold_id)
  {
    if ((dim == 2) && (spacedim == 2))
      internal::set_manifold_id<2,2>(manifold_id, cell_accessor);
    else if ((dim == 2) && (spacedim == 3))
      internal::set_manifold_id<2,3>(manifold_id, cell_accessor);
    else
      internal::set_manifold_id<3,3>(manifold_id, cell_accessor);
  }



  int CellAccessorWrapper::get_manifold_id() const
  {
    if ((dim == 2) && (spacedim == 2))
      return internal::get_manifold_id<2,2>(cell_accessor);
    else if ((dim== 2) && (spacedim == 3))
      return internal::get_manifold_id<2,3>(cell_accessor);
    else
      return internal::get_manifold_id<3,3>(cell_accessor);
  }
}

DEAL_II_NAMESPACE_CLOSE
