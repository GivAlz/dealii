// ---------------------------------------------------------------------
//
// Copyright (C) 2007 - 2018 by the deal.II authors
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

#include <deal.II/lac/filtered_matrix.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

#include "../tests.h"

template <typename number>
void
checkApply_Constraints(FullMatrix<number> &A,
                       Vector<number> &    V,
                       bool                matrix_is_symmetric = false)
{
  deallog << "apply_constraints" << std::endl;

  FilteredMatrix<Vector<double>> F;
  F.initialize(A);
  F.add_constraint(0, 1);

  F.apply_constraints(V, matrix_is_symmetric);

  for (unsigned int i = 0; i < V.size(); ++i)
    deallog << V(i) << '\t';
  deallog << std::endl;
}

int
main()
{
  std::ofstream logfile("output");
  deallog << std::fixed;
  deallog << std::setprecision(4);
  deallog.attach(logfile);

  const double Adata[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

  FullMatrix<double> A(3, 3);

  A.fill(Adata);

  Vector<double> V1(3);
  Vector<double> V2(3);

  V1(0) = V2(0) = 1;
  V1(1) = V2(1) = 2;
  V1(2) = V2(2) = 3;

  checkApply_Constraints<double>(A, V1, false);
  checkApply_Constraints<double>(A, V2, true);
}
