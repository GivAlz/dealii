//----------------------------------------------------------------------
//    $Id$
//    Version: $Name$ 
//
//    Copyright (C) 2007 by the deal.II authors
//
//    This file is subject to QPL and may not be  distributed
//    without copyright and license information. Please refer
//    to the file deal.II/doc/license.html for the  text  and
//    further information on this license.
//
//----------------------------------------------------------------------

// Refinement base: perform REF/dim refinements
// REF 21 needs 9GBytes on a 64 Bit architecture

#ifdef DEBUG
#define REF 6
#else
#define REF 21
#endif

#include <iomanip>

#include <base/logstream.h>
#include <base/quadrature_lib.h>
#include <grid/tria.h>
#include <grid/grid_generator.h>
#include <dofs/dof_handler.h>
#include <dofs/dof_accessor.h>
#include <fe/mapping_q1.h>
#include <fe/fe_q.h>
#include <fe/fe_dgq.h>
#include <fe/fe_values.h>
#include <fe/fe_system.h>

using namespace dealii;

template <int dim>
void indices (const DoFHandler<dim>& dof)
{
  typedef typename DoFHandler<dim>::active_cell_iterator I;
  
  std::vector<unsigned int> dofs(dof.get_fe().dofs_per_cell);
  const I end = dof.end();

  for (unsigned int k=0;k<10;++k)
    for (I i=dof.begin_active(); i!=end;++i)
      i->get_dof_indices(dofs);
}


template <int dim>
void fevalues (const DoFHandler<dim>& dof,
	       UpdateFlags updates)
{
  typedef typename DoFHandler<dim>::active_cell_iterator I;
  const I end = dof.end();
  
  QGauss<dim> quadrature(5);
  MappingQ1<dim> mapping;
  FEValues<dim> fe(mapping, dof.get_fe(), quadrature, updates);
  
  for (I i=dof.begin_active(); i!=end;++i)
    fe.reinit(i);
}


template <int dim>
void check ()
{
  deallog << "Mesh" << std::endl;
  Triangulation<dim> tr;
  GridGenerator::hyper_cube(tr);
  tr.refine_global(REF/dim);
  
  for (unsigned int i=1;i<5;++i)
    {
      FE_Q<dim> q(i);
      deallog.push(q.get_name());
      deallog << "Dofs per cell " << q.dofs_per_cell << std::endl;
      DoFHandler<dim> dof(tr);
      dof.distribute_dofs(q);
      deallog << "Dofs " << dof.n_dofs() << std::endl;
      indices(dof);
      deallog << "Index" << std::endl;
      fevalues(dof,  update_q_points | update_JxW_values);
      deallog << "qpoints|JxW" << std::endl;
      fevalues(dof,  update_values | update_JxW_values);
      deallog << "values|JxW" << std::endl;
      fevalues(dof,  update_values | update_gradients | update_JxW_values);
      deallog << "values|gradients|JxW" << std::endl;
      deallog.pop();
    }
}


int main()
{
  deallog.log_execution_time(true);
  deallog.log_time_differences(true);
  check<2>();
  check<3>();
}

