// ---------------------------------------------------------------------
//
// Copyright (C) 2012 - 2018 by the deal.II authors
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

#include <deal.II/base/vectorization.h>

#include <type_traits>

#include "../tests.h"

int
main()
{
  initlog();

  if (!std::is_pod<VectorizedArray<double>>::value)
    deallog.get_file_stream()
      << "Not OK because VectorizedArray<double> is not POD!" << std::endl;

  if (!std::is_trivial<VectorizedArray<double>>::value)
    deallog.get_file_stream()
      << "Not OK because VectorizedArray<double> is not trivial!" << std::endl;

  deallog.get_file_stream() << "OK" << std::endl;
}
